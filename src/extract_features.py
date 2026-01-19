"""
Feature extraction script for CLAM pipeline.
Extracts features from tile images using ResNet50 pretrained on ImageNet.
Saves .pt files containing features and coordinates for each patient.
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import re


class ResNet50FeatureExtractor(nn.Module):
    """ResNet50 truncated at the global average pooling layer for feature extraction."""
    
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final FC layer, keep up to avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 2048)
        return x


def parse_tile_coords(filename):
    """
    Parse tile filename to extract y, x coordinates.
    Expected format: y_x.png (e.g., 0_245.png)
    """
    basename = os.path.splitext(filename)[0]
    match = re.match(r'^(\d+)_(\d+)$', basename)
    if match:
        y = int(match.group(1))
        x = int(match.group(2))
        return y, x
    else:
        raise ValueError(f"Cannot parse coordinates from filename: {filename}")


def get_tile_transform():
    """
    Returns the ImageNet normalization transform for tiles.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])


def extract_features_for_patient(patient_dir, model, transform, device, batch_size=32):
    """
    Extract features for all tiles of a single patient.
    
    Args:
        patient_dir: Path to the patient's tile directory
        model: Feature extraction model
        transform: Image preprocessing transform
        device: torch device
        batch_size: Batch size for processing
        
    Returns:
        features: Tensor of shape (N, 2048) where N is number of tiles
        coords: Tensor of shape (N, 2) with (y, x) coordinates
        tile_names: List of tile filenames
    """
    tile_files = [f for f in os.listdir(patient_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if len(tile_files) == 0:
        return None, None, None
    
    all_features = []
    all_coords = []
    tile_names = []
    
    # Process in batches
    images_batch = []
    coords_batch = []
    names_batch = []
    
    for tile_file in tile_files:
        tile_path = os.path.join(patient_dir, tile_file)
        
        try:
            # Parse coordinates from filename
            y, x = parse_tile_coords(tile_file)
            
            # Load and transform image
            img = Image.open(tile_path).convert('RGB')
            img_tensor = transform(img)
            
            images_batch.append(img_tensor)
            coords_batch.append([y, x])
            names_batch.append(tile_file)
            
            # Process batch when full
            if len(images_batch) >= batch_size:
                batch_tensor = torch.stack(images_batch).to(device)
                with torch.no_grad():
                    batch_features = model(batch_tensor)
                all_features.append(batch_features.cpu())
                all_coords.extend(coords_batch)
                tile_names.extend(names_batch)
                
                images_batch = []
                coords_batch = []
                names_batch = []
                
        except Exception as e:
            print(f"Warning: Could not process {tile_file}: {e}")
            continue
    
    # Process remaining tiles
    if len(images_batch) > 0:
        batch_tensor = torch.stack(images_batch).to(device)
        with torch.no_grad():
            batch_features = model(batch_tensor)
        all_features.append(batch_features.cpu())
        all_coords.extend(coords_batch)
        tile_names.extend(names_batch)
    
    if len(all_features) == 0:
        return None, None, None
    
    # Concatenate all features
    features = torch.cat(all_features, dim=0)
    coords = torch.tensor(all_coords, dtype=torch.long)
    
    return features, coords, tile_names


def main():
    parser = argparse.ArgumentParser(description='Extract features from tile images')
    parser.add_argument('--tiles_dir', type=str, default='dataset_tiles',
                        help='Directory containing patient tile folders')
    parser.add_argument('--output_dir', type=str, default='features',
                        help='Directory to save extracted features')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print("Loading ResNet50 feature extractor...")
    model = ResNet50FeatureExtractor()
    model = model.to(device)
    model.eval()
    
    # Get transform
    transform = get_tile_transform()
    
    # Get list of patient directories
    patient_dirs = [d for d in os.listdir(args.tiles_dir) 
                    if os.path.isdir(os.path.join(args.tiles_dir, d))]
    
    print(f"Found {len(patient_dirs)} patients")
    
    # Process each patient
    for patient_id in tqdm(patient_dirs, desc="Extracting features"):
        patient_path = os.path.join(args.tiles_dir, patient_id)
        output_path = os.path.join(args.output_dir, f"{patient_id}.pt")
        
        # Skip if already processed
        if os.path.exists(output_path):
            continue
        
        # Extract features
        features, coords, tile_names = extract_features_for_patient(
            patient_path, model, transform, device, args.batch_size
        )
        
        if features is None:
            print(f"Warning: No valid tiles for patient {patient_id}")
            continue
        
        # Save as .pt file with dictionary structure
        save_dict = {
            'features': features,      # Shape: (N, 2048)
            'coords': coords,          # Shape: (N, 2) - (y, x)
            'tile_names': tile_names   # List of filenames
        }
        
        torch.save(save_dict, output_path)
    
    print(f"\nFeature extraction complete. Files saved to {args.output_dir}/")
    print(f"Feature dimension: 2048 (ResNet50)")


if __name__ == '__main__':
    main()
