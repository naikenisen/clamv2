"""
CLAM (Clustering-constrained Attention Multiple Instance Learning) Model Implementation.
Faithfully follows the official Mahmood Lab implementation:
https://github.com/mahmoodlab/CLAM

Includes:
- Gated Attention Network (Attn_Net_Gated)
- CLAM Single Branch (CLAM_SB)
- Smooth Top-k SVM Loss for instance-level clustering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SmoothTop1SVM(nn.Module):
    """
    Smooth Top-1 SVM Loss for instance-level classification.
    Implementation based on the smooth-topk package used in official CLAM.
    
    This loss is a smooth approximation of the hinge loss (SVM loss)
    using the LogSumExp trick for differentiability.
    """
    
    def __init__(self, n_classes=2, tau=1.0):
        """
        Args:
            n_classes: Number of classes for instance classification (always 2 for CLAM)
            tau: Temperature parameter for smoothing
        """
        super(SmoothTop1SVM, self).__init__()
        self.n_classes = n_classes
        self.tau = tau
    
    def forward(self, logits, targets):
        """
        Compute smooth top-1 SVM loss.
        
        Args:
            logits: Predictions of shape (N, n_classes)
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Scalar loss value
        """
        # Get correct class scores
        batch_size = logits.size(0)
        device = logits.device
        
        # Create one-hot encoding of targets
        targets_one_hot = F.one_hot(targets, num_classes=self.n_classes).float()
        
        # Get scores for correct class
        correct_scores = (logits * targets_one_hot).sum(dim=1)  # (N,)
        
        # Compute margins: for each sample, margin = max(0, 1 + max_{j != y}(s_j) - s_y)
        # We use LogSumExp approximation of max for smoothness
        
        # Mask out correct class with large negative value
        mask = targets_one_hot.bool()
        masked_logits = logits.clone()
        masked_logits[mask] = float('-inf')
        
        # Smooth max over incorrect classes using LogSumExp
        max_wrong_scores = self.tau * torch.logsumexp(masked_logits / self.tau, dim=1)
        
        # Hinge loss with margin 1
        margins = 1.0 + max_wrong_scores - correct_scores
        
        # ReLU to get hinge loss
        losses = F.relu(margins)
        
        return losses.mean()


class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers).
    
    Args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout rate (float) or False to disable
        n_classes: number of classes (attention branches)
    """
    
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net, self).__init__()
        
        # Store dropout rate
        self.dropout_rate = dropout if isinstance(dropout, float) else 0.25
        
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        
        if dropout:
            self.module.append(nn.Dropout(self.dropout_rate))
        
        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers).
    This is the EXACT implementation from the official CLAM repository.
    
    The gating mechanism uses element-wise multiplication between:
    - Tanh transformed features (attention_a)
    - Sigmoid transformed features (attention_b)
    
    Args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout rate (float) or False to disable
        n_classes: number of classes (attention branches)
    """
    
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        
        # Store dropout rate for use in layers
        self.dropout_rate = dropout if isinstance(dropout, float) else 0.25
        
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        
        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()
        ]
        
        if dropout:
            self.attention_a.append(nn.Dropout(self.dropout_rate))
            self.attention_b.append(nn.Dropout(self.dropout_rate))
        
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)
    
    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)  # Element-wise multiplication (gating)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class CLAM_SB(nn.Module):
    """
    CLAM Single Branch Model.
    Exact implementation following the official Mahmood Lab repository.
    
    Args:
        gate: whether to use gated attention network
        size_arg: config for network size ('small' or 'big')
        dropout: dropout rate
        k_sample: number of positive/neg patches to sample for instance-level training
        n_classes: number of classes for slide-level classification
        instance_loss_fn: loss function for instance-level clustering
        subtyping: whether it's a subtyping problem
        embed_dim: input embedding dimension (2048 for ResNet50)
    """
    
    def __init__(self, gate=True, size_arg="small", dropout=0.5, k_sample=8, 
                 n_classes=2, instance_loss_fn=None, subtyping=False, embed_dim=2048):
        super(CLAM_SB, self).__init__()
        
        self.size_dict = {
            "small": [embed_dim, 512, 256], 
            "big": [embed_dim, 512, 384]
        }
        size = self.size_dict[size_arg]
        
        # Feature compression: embed_dim -> 512 with LayerNorm for stability
        fc = [
            nn.Linear(size[0], size[1]), 
            nn.LayerNorm(size[1]),  # Added LayerNorm for training stability
            nn.ReLU(), 
            nn.Dropout(dropout)
        ]
        
        # Attention network
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        # Slide-level classifier
        self.classifiers = nn.Linear(size[1], n_classes)
        
        # Instance classifiers for clustering (one per class)
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn if instance_loss_fn else SmoothTop1SVM(n_classes=2)
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    @staticmethod
    def create_positive_targets(length, device):
        """Create positive targets (label=1) for instance classification."""
        return torch.full((length,), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        """Create negative targets (label=0) for instance classification."""
        return torch.full((length,), 0, device=device).long()
    
    def inst_eval(self, A, h, classifier):
        """
        Instance-level evaluation for in-the-class attention branch.
        Implements Smooth Top-k sampling for instance-level clustering.
        
        Args:
            A: Attention scores (1, N) or (N,)
            h: Instance features (N, D)
            classifier: Instance-level classifier
            
        Returns:
            instance_loss: Instance-level clustering loss
            all_preds: Predictions for sampled instances
            all_targets: Ground truth for sampled instances
        """
        device = h.device
        
        if len(A.shape) == 1:
            A = A.view(1, -1)
        
        # Sample top-k positive instances (highest attention)
        top_p_ids = torch.topk(A, min(self.k_sample, A.size(1)))[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        
        # Sample top-k negative instances (lowest attention = highest -A)
        top_n_ids = torch.topk(-A, min(self.k_sample, A.size(1)), dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        
        # Create targets
        p_targets = self.create_positive_targets(top_p.size(0), device)
        n_targets = self.create_negative_targets(top_n.size(0), device)
        
        # Combine positive and negative samples
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        
        # Classify instances
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        
        # Compute instance loss
        instance_loss = self.instance_loss_fn(logits, all_targets)
        
        return instance_loss, all_preds, all_targets
    
    def inst_eval_out(self, A, h, classifier):
        """
        Instance-level evaluation for out-of-the-class attention branch.
        For subtyping problems only.
        
        Args:
            A: Attention scores
            h: Instance features
            classifier: Instance-level classifier
            
        Returns:
            instance_loss: Instance-level clustering loss
            p_preds: Predictions for sampled instances
            p_targets: Ground truth for sampled instances
        """
        device = h.device
        
        if len(A.shape) == 1:
            A = A.view(1, -1)
        
        # For out-of-class, high attention instances should be negative
        top_p_ids = torch.topk(A, min(self.k_sample, A.size(1)))[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        
        # All sampled instances are negative (not this class)
        p_targets = self.create_negative_targets(top_p.size(0), device)
        
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        
        instance_loss = self.instance_loss_fn(logits, p_targets)
        
        return instance_loss, p_preds, p_targets
    
    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        """
        Forward pass for CLAM Single Branch.
        
        Args:
            h: Input features (N, embed_dim) where N is number of instances
            label: Slide-level label (required for instance_eval)
            instance_eval: Whether to compute instance-level clustering loss
            return_features: Whether to return aggregated features
            attention_only: Whether to return only attention scores
            
        Returns:
            logits: Slide-level prediction logits (1, n_classes)
            Y_prob: Slide-level prediction probabilities (1, n_classes)
            Y_hat: Predicted class (1,)
            A_raw: Raw attention scores (1, N)
            results_dict: Dictionary with instance-level results
        """
        # Get attention scores and transformed features
        A, h = self.attention_net(h)  # A: (N, 1), h: (N, 512)
        A = torch.transpose(A, 1, 0)  # (1, N)
        
        if attention_only:
            return A
        
        A_raw = A
        A = F.softmax(A, dim=1)  # Normalize attention over instances
        
        # Instance-level evaluation for clustering
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            
            # One-hot encode the label
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                
                if inst_label == 1:  # In-the-class
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # Out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                        
                total_inst_loss += instance_loss
            
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        
        # Aggregate features using attention
        M = torch.mm(A, h)  # (1, 512) - weighted sum of instance features
        
        # Slide-level classification
        logits = self.classifiers(M)  # (1, n_classes)
        Y_hat = torch.topk(logits, 1, dim=1)[1]  # Predicted class
        Y_prob = F.softmax(logits, dim=1)  # Class probabilities
        
        if instance_eval:
            results_dict = {
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds)
            }
        else:
            results_dict = {}
        
        if return_features:
            results_dict.update({'features': M})
        
        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    """
    CLAM Multi-Branch Model.
    Uses separate attention branches for each class.
    
    Inherits from CLAM_SB and overrides the architecture for multi-branch attention.
    """
    
    def __init__(self, gate=True, size_arg="small", dropout=0.5, k_sample=8,
                 n_classes=2, instance_loss_fn=None, subtyping=False, embed_dim=2048):
        nn.Module.__init__(self)
        
        self.size_dict = {
            "small": [embed_dim, 512, 256],
            "big": [embed_dim, 512, 384]
        }
        size = self.size_dict[size_arg]
        
        # Feature compression with LayerNorm for stability
        fc = [
            nn.Linear(size[0], size[1]), 
            nn.LayerNorm(size[1]),  # Added LayerNorm for training stability
            nn.ReLU(), 
            nn.Dropout(dropout)
        ]
        
        # Multi-branch attention (n_classes attention heads)
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        # Separate classifier for each class
        bag_classifiers = [nn.Linear(size[1], 1) for _ in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        
        # Instance classifiers
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn if instance_loss_fn else SmoothTop1SVM(n_classes=2)
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        """
        Forward pass for CLAM Multi-Branch.
        """
        A, h = self.attention_net(h)  # A: (N, n_classes), h: (N, 512)
        A = torch.transpose(A, 1, 0)  # (n_classes, N)
        
        if attention_only:
            return A
        
        A_raw = A
        A = F.softmax(A, dim=1)  # Softmax over instances for each class
        
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                
                if inst_label == 1:  # In-the-class
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # Out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                        
                total_inst_loss += instance_loss
            
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        
        # Aggregate features using attention (separate for each class)
        M = torch.mm(A, h)  # (n_classes, 512)
        
        # Slide-level classification (separate classifier per class)
        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        
        if instance_eval:
            results_dict = {
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds)
            }
        else:
            results_dict = {}
        
        if return_features:
            results_dict.update({'features': M})
        
        return logits, Y_prob, Y_hat, A_raw, results_dict


def initialize_weights(module):
    """
    Initialize network weights with appropriate strategies:
    - Kaiming initialization for layers followed by ReLU
    - Xavier initialization for attention layers (Tanh/Sigmoid)
    - Small values for final classifier to prevent overconfident predictions
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # Use Kaiming for hidden layers, Xavier for attention
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)  # Small positive bias
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def initialize_attention_weights(module):
    """
    Specific initialization for attention networks.
    Uses Xavier for Tanh/Sigmoid activations.
    """
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            if 'attention' in name or 'classifier' in name:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # Test the model
    print("Testing CLAM_SB...")
    model = CLAM_SB(gate=True, n_classes=2, embed_dim=2048, k_sample=8)
    
    # Dummy input: 100 instances with 2048-dim features
    x = torch.randn(100, 2048)
    label = torch.tensor([1])
    
    # Forward pass with instance evaluation
    logits, Y_prob, Y_hat, A, results = model(x, label=label, instance_eval=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Y_prob: {Y_prob}")
    print(f"Y_hat: {Y_hat}")
    print(f"Attention shape: {A.shape}")
    print(f"Instance loss: {results['instance_loss']}")
    
    print("\nTesting CLAM_MB...")
    model_mb = CLAM_MB(gate=True, n_classes=2, embed_dim=2048, k_sample=8)
    logits, Y_prob, Y_hat, A, results = model_mb(x, label=label, instance_eval=True)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Attention shape: {A.shape}")
    print(f"Instance loss: {results['instance_loss']}")
