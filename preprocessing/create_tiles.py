"""
Script pour découper les images WSI en tiles de 256x256 pixels.
Applique un masque Otsu pour ne garder que le tissu.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re


def extract_patient_id(filename):
    """
    Extrait l'identifiant du patient du nom de fichier.
    Ex: '13901.png' -> '13901'
        '13901_2.png' -> '13901'
        '13901_3.png' -> '13901'
    """
    # Enlever l'extension
    name = Path(filename).stem
    
    # Chercher le pattern: nombre suivi optionnellement de _nombre
    match = re.match(r'^(\d+)(?:_\d+)?$', name)
    if match:
        return match.group(1)
    
    # Si pas de match, retourner le nom complet (sans extension)
    return name


def apply_otsu_mask(image):
    """
    Applique un masque d'Otsu pour isoler le tissu.
    Retourne le masque binaire où True = tissu à conserver.
    """
    # Convertir en niveaux de gris si nécessaire
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Appliquer le seuil d'Otsu
    # Otsu retourne le seuil et l'image binaire
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Inverser si nécessaire (dépend si le tissu est clair ou foncé)
    # On garde les pixels sombres (tissu) si la moyenne du masque > 127
    if np.mean(binary_mask) > 127:
        binary_mask = cv2.bitwise_not(binary_mask)
    
    return binary_mask > 0


def has_tissue(tile, mask_tile, tissue_threshold=0.5):
    """
    Vérifie si le tile contient suffisamment de tissu.
    
    Args:
        tile: Le tile d'image
        mask_tile: Le masque correspondant
        tissue_threshold: Proportion minimale de tissu (0.0 à 1.0)
    
    Returns:
        True si le tile contient suffisamment de tissu
    """
    tissue_ratio = np.sum(mask_tile) / mask_tile.size
    return tissue_ratio >= tissue_threshold


def create_tiles(image_path, output_base_dir, tile_size=256, tissue_threshold=0.5):
    """
    Découpe une image en tiles de taille fixe avec masque Otsu.
    
    Args:
        image_path: Chemin vers l'image WSI
        output_base_dir: Dossier de sortie de base
        tile_size: Taille des tiles (256x256 par défaut)
        tissue_threshold: Seuil de tissu minimum pour garder un tile
    
    Returns:
        Nombre de tiles créés
    """
    # Charger l'image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Erreur: impossible de charger {image_path}")
        return 0
    
    h, w = image.shape[:2]
    
    # Appliquer le masque d'Otsu
    tissue_mask = apply_otsu_mask(image)
    
    # Extraire le nom du fichier et l'ID patient
    filename = Path(image_path).name
    image_name = Path(image_path).stem
    patient_id = extract_patient_id(filename)
    
    # Créer le dossier de sortie pour ce patient
    patient_dir = Path(output_base_dir) / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    # Découper en tiles sans overlapping
    tile_count = 0
    
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            # Extraire le tile
            tile = image[y:y+tile_size, x:x+tile_size]
            mask_tile = tissue_mask[y:y+tile_size, x:x+tile_size]
            
            # Vérifier si le tile contient suffisamment de tissu
            if has_tissue(tile, mask_tile, tissue_threshold):
                tile_count += 1
                
                # Sauvegarder le tile
                tile_filename = f"{image_name}_{tile_count}.png"
                tile_path = patient_dir / tile_filename
                cv2.imwrite(str(tile_path), tile)
    
    return tile_count


def process_dataset(dataset_dir, output_dir, tile_size=256, tissue_threshold=0.5):
    """
    Traite toutes les images du dataset.
    
    Args:
        dataset_dir: Dossier contenant les images WSI
        output_dir: Dossier de sortie pour les tiles
        tile_size: Taille des tiles
        tissue_threshold: Seuil de tissu minimum
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    # Créer le dossier de sortie
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lister toutes les images
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.svs']
    images = []
    for ext in image_extensions:
        images.extend(dataset_path.glob(f'*{ext}'))
        images.extend(dataset_path.glob(f'*{ext.upper()}'))
    
    if not images:
        print(f"Aucune image trouvée dans {dataset_dir}")
        return
    
    print(f"Traitement de {len(images)} images...")
    print(f"Taille des tiles: {tile_size}x{tile_size} pixels")
    print(f"Seuil de tissu: {tissue_threshold * 100}%")
    print(f"Dossier de sortie: {output_dir}")
    print()
    
    total_tiles = 0
    
    # Traiter chaque image avec barre de progression
    for image_path in tqdm(images, desc="Traitement des WSI", unit="image"):
        try:
            tiles_created = create_tiles(
                image_path,
                output_path,
                tile_size=tile_size,
                tissue_threshold=tissue_threshold
            )
            total_tiles += tiles_created
            tqdm.write(f"  {image_path.name}: {tiles_created} tiles créés")
        except Exception as e:
            tqdm.write(f"  Erreur avec {image_path.name}: {str(e)}")
    
    print()
    print(f"Terminé! Total: {total_tiles} tiles créés")


if __name__ == "__main__":
    # Configuration
    DATASET_DIR = "dataset"
    OUTPUT_DIR = "dataset_tiles"
    TILE_SIZE = 256
    TISSUE_THRESHOLD = 0.5  # 50% minimum de tissu dans le tile
    
    # Lancer le traitement
    process_dataset(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        tile_size=TILE_SIZE,
        tissue_threshold=TISSUE_THRESHOLD
    )
