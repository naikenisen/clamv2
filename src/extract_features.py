"""
Feature extraction script for CLAM pipeline using Phikon.

Phikon: Self-Supervised Vision Transformer for Pathology (Owkin)
- Architecture: ViT-B/16 trained with DINOv2 on TCGA
- Output dimension: 768
- Input size: 224x224
- Publicly available: https://huggingface.co/owkin/phikon

Reference: Filiot et al., "Scaling Self-Supervised Learning for 
Histopathology with Masked Image Modeling", MedRxiv 2023
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoImageProcessor
except ImportError:
    raise ImportError(
        "transformers is required for Phikon.\n"
        "Install with: pip install transformers huggingface_hub"
    )


class PhikonFeatureExtractor(nn.Module):
    """
    Phikon feature extractor for histopathology tiles.
    Uses ViT-B/16 pretrained on TCGA with DINOv2.
    """
    
    def __init__(self, device='cuda'):
        super(PhikonFeatureExtractor, self).__init__()
        
        print("Loading Phikon model from HuggingFace (owkin/phikon)...")
        
        self.model = AutoModel.from_pretrained("owkin/phikon", trust_remote_code=True)
        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon", trust_remote_code=True)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.embed_dim = 768
        
        # Phikon transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.processor.image_mean,
                std=self.processor.image_std
            )
        ])
        
        print(f"Phikon loaded successfully. Embedding dimension: {self.embed_dim}")
        
    def forward(self, x):
        with torch.no_grad():
            outputs = self.model(x)
            # Use CLS token as global representation
            features = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        return features
    
    def get_transform(self):
        return self.transform


class TileDataset(torch.utils.data.Dataset):
    """Dataset for loading tiles from a slide directory."""
    
    VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    
    def __init__(self, tiles_dir, transform=None):
        self.tiles_dir = tiles_dir
        self.transform = transform
        
        # Get all valid image files
        self.tile_paths = []
        for f in os.listdir(tiles_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in self.VALID_EXTENSIONS:
                self.tile_paths.append(os.path.join(tiles_dir, f))
        
        # Sort for reproducibility
        self.tile_paths.sort()
        
    def __len__(self):
        return len(self.tile_paths)
    
    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        try:
            image = Image.open(tile_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, tile_path
        except Exception as e:
            print(f"Error loading {tile_path}: {e}")
            # Return a blank image on error
            if self.transform:
                blank = Image.new('RGB', (224, 224), (255, 255, 255))
                return self.transform(blank), tile_path
            return None, tile_path


def extract_features_for_slide(
    model: PhikonFeatureExtractor,
    tiles_dir: str,
    output_path: str,
    batch_size: int = 64,
    num_workers: int = 4
):
    """
    Extract features for all tiles in a slide directory.
    
    Args:
        model: Phikon feature extractor
        tiles_dir: Directory containing tile images
        output_path: Path to save the .pt features file
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
    
    Returns:
        Tensor of shape (num_tiles, 768)
    """
    dataset = TileDataset(tiles_dir, transform=model.get_transform())
    
    if len(dataset) == 0:
        print(f"  Warning: No tiles found in {tiles_dir}")
        return None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    all_features = []
    
    for batch, _ in dataloader:
        batch = batch.to(model.device)
        features = model(batch)
        all_features.append(features.cpu())
    
    # Concatenate all features
    features_tensor = torch.cat(all_features, dim=0)  # (N, 768)
    
    # Save features
    torch.save(features_tensor, output_path)
    
    return features_tensor


def main():
    parser = argparse.ArgumentParser(
        description='Extract Phikon features from histopathology tiles'
    )
    parser.add_argument(
        '--tiles_dir', 
        type=str, 
        default='dataset_tiles',
        help='Directory containing slide subdirectories with tiles'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='features',
        help='Directory to save extracted features'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64,
        help='Batch size for feature extraction'
    )
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PHIKON FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Tiles directory: {args.tiles_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Load model
    model = PhikonFeatureExtractor(device=args.device)
    
    # Get all slide directories
    slide_dirs = []
    for name in os.listdir(args.tiles_dir):
        slide_path = os.path.join(args.tiles_dir, name)
        if os.path.isdir(slide_path):
            slide_dirs.append((name, slide_path))
    
    slide_dirs.sort()
    print(f"\nFound {len(slide_dirs)} slides to process")
    
    # Process each slide
    success_count = 0
    error_count = 0
    
    for slide_id, slide_path in tqdm(slide_dirs, desc="Extracting features"):
        output_path = os.path.join(args.output_dir, f"{slide_id}.pt")
        
        # Skip if already processed
        if os.path.exists(output_path):
            tqdm.write(f"  Skipping {slide_id} (already exists)")
            success_count += 1
            continue
        
        try:
            features = extract_features_for_slide(
                model=model,
                tiles_dir=slide_path,
                output_path=output_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            if features is not None:
                tqdm.write(f"  {slide_id}: {features.shape[0]} tiles -> {output_path}")
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            tqdm.write(f"  Error processing {slide_id}: {e}")
            error_count += 1
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {success_count} slides")
    print(f"Errors: {error_count} slides")
    print(f"Feature dimension: 768")
    print(f"Features saved to: {args.output_dir}")
    print("\nNext step: Update train.sh with --embed_dim 768")
    print("=" * 60)


if __name__ == '__main__':
    main()
