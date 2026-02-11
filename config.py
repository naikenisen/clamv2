import os
from datetime import datetime

# =========================================================================
# HYPERPARAMETERS (Fixed Best Configuration)
# =========================================================================
lr = 0.0001
dropout = 0.4
k_sample = 12
bag_weight = 0.8

# =========================================================================
# DATA & PATHS
# =========================================================================
clinical_csv = 'clinical_data.csv'
features_dir = 'features'
output_dir = f"results_{datetime.now().strftime('%Y-%m-%d')}"

# =========================================================================
# MODEL SETTINGS
# =========================================================================
model_type = 'clam_sb'  # or 'clam_mb'
model_size = 'small'
embed_dim = 768
gate = True
subtyping = False

# =========================================================================
# TRAINING SETTINGS
# =========================================================================
seed = 42
max_epochs = 100
warmup_epochs = 5
patience = 15          # Early stopping patience
stop_epoch = 15        # Minimum epochs before early stopping
weight_decay = 1e-4
max_grad_norm = 1.0
bag_dropout = 0.15     # Dropout applied to instances within a bag during training
min_lr = 1e-6
num_workers = 4
device = 'cuda'        # 'cuda' or 'cpu'

# =========================================================================
# LOSS SETTINGS
# =========================================================================
use_focal_loss = False
focal_gamma = 2.0
label_smoothing = 0.1

# =========================================================================
# DATA SPLIT
# =========================================================================
test_size = 0.15       # Percentage of data for Final Testing
val_size = 0.15        # Percentage of Train+Val data used for Validation (Early Stopping)