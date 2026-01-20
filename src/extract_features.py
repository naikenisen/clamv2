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


def parse_tile_coordinates(filename):
    """
    Parse tile coordinates from filename.
    
    Expected format: {x}_{y}.{ext} where x and y are pixel coordinates.
    Examples: 
        '1024_1280.png' -> (1024, 1280)
        '0_768.png' -> (0, 768)
    
    Args:
        filename: Tile filename (e.g., '1024_1280.png')
        
    Returns:
        Tuple (x, y) of pixel coordinates, or None if parsing fails
    """
    try:
        # Remove extension and split by underscore
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        
        if len(parts) >= 2:
            x = int(parts[0])
            y = int(parts[1])
            return (x, y)
        else:
            return None
    except (ValueError, IndexError):
        return None


class TileDataset(torch.utils.data.Dataset):
    """
    Dataset for loading tiles from a slide directory.
    Extracts coordinates from tile filenames in format {x}_{y}.{ext}.
    """
    
    VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    
    def __init__(self, tiles_dir, transform=None, tile_size=256):
        """
        Args:
            tiles_dir: Directory containing tile images
            transform: Optional transforms to apply
            tile_size: Size of each tile in pixels (default: 256)
        """
        self.tiles_dir = tiles_dir
        self.transform = transform
        self.tile_size = tile_size
        
        # Get all valid image files with their coordinates
        self.tile_paths = []
        self.tile_names = []
        self.coordinates = []
        
        for f in os.listdir(tiles_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in self.VALID_EXTENSIONS:
                coords = parse_tile_coordinates(f)
                if coords is not None:
                    self.tile_paths.append(os.path.join(tiles_dir, f))
                    self.tile_names.append(f)
                    self.coordinates.append(coords)
                else:
                    print(f"  Warning: Could not parse coordinates from '{f}', skipping")
        
        # Sort by coordinates for reproducibility (y first, then x for row-major order)
        sorted_indices = sorted(
            range(len(self.coordinates)), 
            key=lambda i: (self.coordinates[i][1], self.coordinates[i][0])
        )
        self.tile_paths = [self.tile_paths[i] for i in sorted_indices]
        self.tile_names = [self.tile_names[i] for i in sorted_indices]
        self.coordinates = [self.coordinates[i] for i in sorted_indices]
        
    def __len__(self):
        return len(self.tile_paths)
    
    def get_coordinates(self):
        """Return all tile coordinates as a tensor (N, 2) in (x, y) format."""
        return torch.tensor(self.coordinates, dtype=torch.long)
    
    def get_tile_names(self):
        """Return list of tile filenames."""
        return self.tile_names
    
    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        tile_name = self.tile_names[idx]
        coords = self.coordinates[idx]
        
        try:
            image = Image.open(tile_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, tile_name, coords
        except Exception as e:
            print(f"Error loading {tile_path}: {e}")
            # Return a blank image on error
            if self.transform:
                blank = Image.new('RGB', (224, 224), (255, 255, 255))
                return self.transform(blank), tile_name, coords
            return None, tile_name, coords


def extract_features_for_slide(
    model: PhikonFeatureExtractor,
    tiles_dir: str,
    output_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    tile_size: int = 256
):
    """
    Extract features for all tiles in a slide directory.
    
    Saves a dictionary with:
        - 'features': Tensor of shape (num_tiles, 768)
        - 'coords': Tensor of shape (num_tiles, 2) with (x, y) pixel coordinates
        - 'tile_names': List of tile filenames
        - 'tile_size': Size of each tile in pixels
        - 'embed_dim': Feature embedding dimension (768 for Phikon)
    
    Args:
        model: Phikon feature extractor
        tiles_dir: Directory containing tile images
        output_path: Path to save the .pt features file
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        tile_size: Size of each tile in pixels (default: 256)
    
    Returns:
        Dictionary with features, coords, tile_names, or None if no tiles found
    """
    dataset = TileDataset(tiles_dir, transform=model.get_transform(), tile_size=tile_size)
    
    if len(dataset) == 0:
        print(f"  Warning: No tiles found in {tiles_dir}")
        return None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: maintain order for coordinate alignment
        num_workers=num_workers,
        pin_memory=True
    )
    
    all_features = []
    
    for batch_images, batch_names, batch_coords in dataloader:
        batch_images = batch_images.to(model.device)
        features = model(batch_images)
        all_features.append(features.cpu())
    
    # Concatenate all features
    features_tensor = torch.cat(all_features, dim=0)  # (N, 768)
    
    # Get coordinates and tile names from dataset (maintains order)
    coords_tensor = dataset.get_coordinates()  # (N, 2)
    tile_names = dataset.get_tile_names()
    
    # Validate alignment
    assert features_tensor.shape[0] == coords_tensor.shape[0] == len(tile_names), \
        f"Mismatch: features={features_tensor.shape[0]}, coords={coords_tensor.shape[0]}, names={len(tile_names)}"
    
    # Create output dictionary
    output_data = {
        'features': features_tensor,      # (N, 768)
        'coords': coords_tensor,          # (N, 2) in (x, y) pixel coordinates
        'tile_names': tile_names,         # List of filenames
        'tile_size': tile_size,           # Tile size in pixels
        'embed_dim': model.embed_dim,     # Feature dimension (768)
    }
    
    # Save features with metadata
    torch.save(output_data, output_path)
    
    return output_data


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
    parser.add_argument(
        '--tile_size',
        type=int,
        default=256,
        help='Tile size in pixels (default: 256)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PHIKON FEATURE EXTRACTION WITH COORDINATES")
    print("=" * 60)
    print(f"Tiles directory: {args.tiles_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Tile size: {args.tile_size}px")
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
            result = extract_features_for_slide(
                model=model,
                tiles_dir=slide_path,
                output_path=output_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                tile_size=args.tile_size
            )
            
            if result is not None:
                n_tiles = result['features'].shape[0]
                tqdm.write(f"  {slide_id}: {n_tiles} tiles, coords shape {result['coords'].shape} -> {output_path}")
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
    print(f"Tile size: {args.tile_size}px")
    print(f"Features saved to: {args.output_dir}")
    print("\nOutput format (.pt files):")
    print("  - 'features': Tensor (N, 768)")
    print("  - 'coords': Tensor (N, 2) with (x, y) pixel coordinates")
    print("  - 'tile_names': List of tile filenames")
    print("  - 'tile_size': Tile size in pixels")
    print("  - 'embed_dim': Feature dimension (768)")
    print("\nNext step: Update train.sh with --embed_dim 768")
    print("=" * 60)


if __name__ == '__main__':
    main()
