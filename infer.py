"""
Inference script for CLAM model.
Loads trained model and generates attention heatmaps for visualization.
Uses the exact same test split as training for reproducibility.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import CLAM_SB, CLAM_MB, SmoothTop1SVM
from src.data_loader import get_test_loader_from_splits, load_splits


def load_model(model_path, model_type='clam_sb', n_classes=2, embed_dim=2048, 
               model_size='small', dropout=0.25, k_sample=8, device='cuda'):
    """
    Load trained CLAM model.
    
    Args:
        model_path: Path to model.pth file
        model_type: Type of CLAM model ('clam_sb' or 'clam_mb')
        n_classes: Number of classes
        embed_dim: Feature embedding dimension
        model_size: Model size ('small' or 'big')
        dropout: Dropout rate
        k_sample: Number of samples for instance clustering
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    model_dict = {
        'gate': True,
        'size_arg': model_size,
        'dropout': dropout,
        'k_sample': k_sample,
        'n_classes': n_classes,
        'instance_loss_fn': SmoothTop1SVM(n_classes=2),
        'subtyping': False,
        'embed_dim': embed_dim
    }
    
    if model_type == 'clam_sb':
        model = CLAM_SB(**model_dict)
    else:
        model = CLAM_MB(**model_dict)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Handle potential 'module.' prefix from DataParallel
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        # Skip instance_loss_fn parameters if present
        if 'instance_loss_fn' not in new_key:
            new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def get_attention_scores(model, features, device):
    """
    Extract attention scores from model.
    
    Args:
        model: CLAM model
        features: Input features (N, embed_dim)
        device: Computation device
        
    Returns:
        attention: Normalized attention scores (N,)
        Y_prob: Class probabilities
        Y_hat: Predicted class
    """
    features = features.to(device)
    
    with torch.no_grad():
        logits, Y_prob, Y_hat, A_raw, _ = model(features, instance_eval=False)
        
        # For multi-branch, select attention for predicted class
        if isinstance(model, CLAM_MB):
            A = A_raw[Y_hat.item()]
        else:
            A = A_raw
        
        # Softmax normalization
        A = F.softmax(A, dim=1)
        A = A.squeeze().cpu().numpy()
    
    return A, Y_prob.cpu().numpy(), Y_hat.item()


def normalize_attention_minmax(attention):
    """
    Min-Max normalize attention scores to [0, 1].
    
    Args:
        attention: Raw attention scores (N,)
        
    Returns:
        Normalized attention scores (N,)
    """
    min_val = attention.min()
    max_val = attention.max()
    
    if max_val - min_val > 0:
        normalized = (attention - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(attention)
    
    return normalized


def create_attention_heatmap(attention_scores, coords, tile_size=256, 
                             output_path=None, colormap='jet'):
    """
    Create a 2D attention heatmap from attention scores and tile coordinates.
    
    Args:
        attention_scores: Normalized attention scores (N,)
        coords: Tile coordinates (N, 2) in (x, y) pixel format
        tile_size: Size of each tile in pixels (for converting to tile indices)
        output_path: Optional path to save the heatmap
        colormap: Matplotlib colormap to use
        
    Returns:
        heatmap: 2D numpy array with attention values
        (x_min, x_max, y_min, y_max): Bounding box in pixel coordinates
    """
    if len(coords) == 0:
        return None, None
    
    # Coords are in (x, y) pixel format - convert to tile indices
    x_coords = coords[:, 0] // tile_size
    y_coords = coords[:, 1] // tile_size
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Normalize tile indices to start from 0
    x_normalized = x_coords - x_min
    y_normalized = y_coords - y_min
    
    # Create heatmap matrix (rows = y, cols = x)
    height = int(y_max - y_min + 1)
    width = int(x_max - x_min + 1)
    heatmap = np.zeros((height, width))
    
    # Fill in attention values (heatmap[row, col] = heatmap[y, x])
    for i, (x, y) in enumerate(zip(x_normalized, y_normalized)):
        heatmap[int(y), int(x)] = attention_scores[i]
    
    if output_path:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        im = ax.imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Attention Score')
        
        ax.set_title('Attention Heatmap')
        ax.set_xlabel('X Tile Index')
        ax.set_ylabel('Y Tile Index')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return heatmap, (y_min, y_max, x_min, x_max)


def create_full_resolution_heatmap(attention_scores, coords, tile_size=256,
                                   output_path=None, colormap='jet',
                                   background_color=(255, 255, 255)):
    """
    Create a full-resolution heatmap where each tile is represented as a tile_size x tile_size block.
    
    Args:
        attention_scores: Normalized attention scores (N,)
        coords: Tile coordinates (N, 2) in (x, y) pixel format
        tile_size: Size of each tile in pixels
        output_path: Optional path to save the heatmap
        colormap: Matplotlib colormap
        background_color: Background color for empty tiles
        
    Returns:
        heatmap_img: PIL Image of the heatmap
    """
    if len(coords) == 0:
        return None
    
    # Coords are in (x, y) pixel format
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Normalize to start from 0
    x_normalized = x_coords - x_min
    y_normalized = y_coords - y_min
    
    # Calculate image size (in pixels)
    width = int(x_max - x_min + tile_size)
    height = int(y_max - y_min + tile_size)
    
    # Create RGBA image with background
    heatmap_img = Image.new('RGBA', (width, height), (*background_color, 255))
    
    # Get colormap
    cmap = plt.cm.get_cmap(colormap)
    
    # Fill in tiles at their actual pixel positions
    for i, (x, y) in enumerate(zip(x_normalized, y_normalized)):
        # Get color for this attention score
        color = cmap(attention_scores[i])
        color_rgb = tuple(int(c * 255) for c in color[:3])
        
        # x, y are already in pixel coordinates (relative to min)
        px = int(x)
        py = int(y)
        
        # Create tile
        tile = Image.new('RGB', (tile_size, tile_size), color_rgb)
        heatmap_img.paste(tile, (px, py))
    
    if output_path:
        heatmap_img.save(output_path)
    
    return heatmap_img


def overlay_heatmap_on_wsi(wsi_path, attention_scores, coords, tile_size=256,
                           output_path=None, alpha=0.5, colormap='jet'):
    """
    Overlay attention heatmap on the original WSI.
    
    Args:
        wsi_path: Path to the WSI image
        attention_scores: Normalized attention scores
        coords: Tile coordinates (N, 2) in (x, y) pixel format
        tile_size: Size of each tile
        output_path: Path to save the overlay
        alpha: Transparency of the heatmap overlay
        colormap: Colormap to use
        
    Returns:
        overlay_img: PIL Image with overlay
    """
    # Load WSI
    wsi = Image.open(wsi_path).convert('RGB')
    wsi_width, wsi_height = wsi.size
    
    # Create heatmap image
    heatmap = create_full_resolution_heatmap(
        attention_scores, coords, tile_size, colormap=colormap
    )
    
    if heatmap is None:
        return None
    
    # Resize heatmap to match WSI size if needed
    heatmap = heatmap.convert('RGB')
    
    # Calculate offset based on coordinate minimums (coords are in x, y format)
    x_min = int(coords[:, 0].min())
    y_min = int(coords[:, 1].min())
    offset_x = x_min  # Already in pixels
    offset_y = y_min  # Already in pixels
    
    # Create overlay at full WSI size
    overlay = Image.new('RGB', (wsi_width, wsi_height), (255, 255, 255))
    overlay.paste(heatmap, (offset_x, offset_y))
    
    # Blend with original WSI
    overlay_img = Image.blend(wsi, overlay, alpha)
    
    if output_path:
        overlay_img.save(output_path)
    
    return overlay_img


def infer_single_patient(model, features, coords, device, tile_names=None):
    """
    Run inference on a single patient.
    
    Args:
        model: CLAM model
        features: Feature tensor (N, embed_dim)
        coords: Coordinate tensor (N, 2)
        device: Computation device
        tile_names: Optional list of tile filenames
        
    Returns:
        Dictionary with prediction results and attention info
    """
    # Get attention scores
    attention, Y_prob, Y_hat = get_attention_scores(model, features, device)
    
    # Normalize attention
    attention_normalized = normalize_attention_minmax(attention)
    
    # Class names for binary classification
    class_names = ['Responder', 'Progressor']
    
    result = {
        'prediction': Y_hat,
        'prediction_label': class_names[Y_hat],
        'probabilities': Y_prob.flatten().tolist(),
        'attention_raw': attention.tolist(),
        'attention_normalized': attention_normalized.tolist(),
        'coords': coords.numpy().tolist(),
        'tile_names': tile_names if tile_names else []
    }
    
    return result


def run_inference(args):
    """
    Main inference function.
    """
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load model configuration if available
    config_path = os.path.join(args.results_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_type = config.get('model_type', args.model_type)
        n_classes = config.get('n_classes', args.n_classes)
        embed_dim = config.get('embed_dim', args.embed_dim)
        model_size = config.get('model_size', args.model_size)
        dropout = config.get('dropout', args.dropout)
        k_sample = config.get('k_sample', args.k_sample)
        print(f"Loaded configuration from {config_path}")
    else:
        model_type = args.model_type
        n_classes = args.n_classes
        embed_dim = args.embed_dim
        model_size = args.model_size
        dropout = args.dropout
        k_sample = args.k_sample
    
    # Load model
    model_path = os.path.join(args.results_dir, 'model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"\nLoading model from {model_path}")
    model = load_model(
        model_path, model_type, n_classes, embed_dim,
        model_size, dropout, k_sample, device
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print("\nLoading test data...")
    if args.patient_id:
        # Single patient inference
        feature_path = os.path.join(args.features_dir, f"{args.patient_id}.pt")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Features not found for patient {args.patient_id}")
        
        data = torch.load(feature_path, weights_only=False)
        
        # Handle both new format (dict) and legacy format (tensor only)
        if isinstance(data, dict):
            features = data['features']
            coords = data['coords']
            tile_names = data.get('tile_names', [])
            tile_size = data.get('tile_size', 256)
        else:
            # Legacy format: data is just the features tensor
            features = data
            coords = torch.zeros((features.shape[0], 2), dtype=torch.long)
            tile_names = []
            tile_size = 256
            print(f"  Warning: Using legacy format without coordinates. Heatmaps will not be spatially accurate.")
        
        print(f"Processing patient {args.patient_id}...")
        print(f"  Features: {features.shape}, Coords: {coords.shape}, Tile size: {tile_size}px")
        result = infer_single_patient(model, features, coords, device, tile_names)
        
        print(f"\nPrediction: {result['prediction_label']}")
        print(f"Probabilities: Responder={result['probabilities'][0]:.4f}, "
              f"Progressor={result['probabilities'][1]:.4f}")
        
        # Create heatmap (tile index grid)
        heatmap_path = os.path.join(args.output_dir, f"{args.patient_id}_heatmap.png")
        create_attention_heatmap(
            np.array(result['attention_normalized']),
            np.array(result['coords']),
            tile_size=tile_size,
            output_path=heatmap_path
        )
        print(f"Heatmap saved to {heatmap_path}")
        
        # Create full resolution heatmap
        if args.full_resolution:
            full_heatmap_path = os.path.join(args.output_dir, f"{args.patient_id}_heatmap_full.png")
            create_full_resolution_heatmap(
                np.array(result['attention_normalized']),
                np.array(result['coords']),
                tile_size=tile_size,
                output_path=full_heatmap_path
            )
            print(f"Full resolution heatmap saved to {full_heatmap_path}")
        
        # Save results
        result_path = os.path.join(args.output_dir, f"{args.patient_id}_results.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {result_path}")
        
    else:
        # Batch inference on test set
        test_loader, test_patient_ids = get_test_loader_from_splits(
            args.clinical_csv,
            args.features_dir,
            splits_path=args.splits_path,
            batch_size=1,
            num_workers=args.num_workers
        )
        
        print(f"Found {len(test_patient_ids)} patients in test set")
        
        all_results = {}
        predictions = []
        labels = []
        
        for batch in tqdm(test_loader, desc="Running inference"):
            features = batch['features']
            coords = batch['coords']
            label = batch['label']
            patient_ids = batch['patient_id']
            tile_names = batch['tile_names']
            
            if isinstance(features, list):
                features = features[0]
                coords = coords[0]
                label = label[0]
                tile_names = tile_names[0] if tile_names else []
            
            patient_id = patient_ids[0]
            
            # Run inference
            result = infer_single_patient(model, features, coords, device, tile_names)
            result['true_label'] = label.item()
            
            all_results[patient_id] = result
            predictions.append(result['prediction'])
            labels.append(label.item())
            
            # Create heatmap for each patient
            if args.generate_heatmaps:
                patient_output_dir = os.path.join(args.output_dir, 'heatmaps')
                os.makedirs(patient_output_dir, exist_ok=True)
                
                heatmap_path = os.path.join(patient_output_dir, f"{patient_id}_heatmap.png")
                create_attention_heatmap(
                    np.array(result['attention_normalized']),
                    np.array(result['coords']),
                    output_path=heatmap_path
                )
        
        # Calculate metrics
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        accuracy = (predictions == labels).mean()
        
        from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
        
        # Get probabilities for AUC
        probs = np.array([all_results[pid]['probabilities'][1] for pid in test_patient_ids 
                         if pid in all_results])
        true_labels = np.array([all_results[pid]['true_label'] for pid in test_patient_ids 
                               if pid in all_results])
        
        if len(np.unique(true_labels)) > 1:
            auc = roc_auc_score(true_labels, probs)
        else:
            auc = float('nan')
        
        print("\n" + "="*60)
        print("Test Set Results")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(labels, predictions))
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=['Responder', 'Progressor']))
        
        # Save summary results
        summary = {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'n_patients': len(predictions),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }
        
        summary_path = os.path.join(args.output_dir, 'inference_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        results_path = os.path.join(args.output_dir, 'inference_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='CLAM Inference and Heatmap Generation')
    
    # Input paths
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing trained model and config')
    parser.add_argument('--features_dir', type=str, default='features',
                        help='Directory containing extracted features')
    parser.add_argument('--clinical_csv', type=str, default='clinical_data.csv',
                        help='Path to clinical data CSV')
    parser.add_argument('--splits_path', type=str, default='splits.json',
                        help='Path to splits JSON file')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='inference_output',
                        help='Directory to save inference results')
    
    # Patient selection
    parser.add_argument('--patient_id', type=str, default=None,
                        help='Specific patient ID to process (if not specified, process all test set)')
    
    # Model parameters (used if config.json not found)
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb'],
                        default='clam_sb', help='Model type')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--embed_dim', type=int, default=2048,
                        help='Feature embedding dimension')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'],
                        default='small', help='Model size')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout rate')
    parser.add_argument('--k_sample', type=int, default=8,
                        help='K sample for instance clustering')
    
    # Heatmap options
    parser.add_argument('--generate_heatmaps', action='store_true', default=True,
                        help='Generate heatmaps for all patients')
    parser.add_argument('--full_resolution', action='store_true',
                        help='Generate full resolution heatmaps')
    parser.add_argument('--colormap', type=str, default='jet',
                        help='Colormap for heatmaps')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    run_inference(args)


if __name__ == '__main__':
    main()
