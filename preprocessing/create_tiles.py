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
import matplotlib.pyplot as plt


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


def apply_otsu_mask(image, morphology_kernel_size=0, gaussian_blur_size=11):
    """
    Applique un masque d'Otsu pour isoler le tissu.
    Retourne le masque binaire où True = tissu à conserver.
    
    Args:
        image: L'image à traiter
        morphology_kernel_size: Taille du kernel pour les opérations morphologiques.
                               Plus grand = comble plus de trous (0 = pas d'opérations morphologiques)
        gaussian_blur_size: Taille du kernel pour le flou gaussien (doit être impair, 0 = pas de flou).
                           Appliqué AVANT Otsu pour lisser l'image et réduire le bruit.
    """
    # Convertir en niveaux de gris si nécessaire
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Appliquer un flou gaussien pour lisser l'image avant Otsu
    if gaussian_blur_size > 0:
        # S'assurer que la taille est impaire
        if gaussian_blur_size % 2 == 0:
            gaussian_blur_size += 1
        gray = cv2.GaussianBlur(gray, (gaussian_blur_size, gaussian_blur_size), 0)
    
    # Appliquer le seuil d'Otsu
    # Otsu retourne le seuil et l'image binaire
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Inverser pour garder le tissu (pixels sombres) et supprimer l'arrière-plan (pixels clairs)
    # Si la moyenne < 127, le masque a déjà les pixels sombres à 0, donc on inverse
    if np.mean(binary_mask) < 127:
        binary_mask = cv2.bitwise_not(binary_mask)
    
    # Appliquer des opérations morphologiques pour combler les petits trous
    if morphology_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (morphology_kernel_size, morphology_kernel_size))
        # Closing: ferme les petits trous dans le tissu
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        # Opening: enlève les petits points de bruit
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    return binary_mask > 0


def validate_mask_interactive(image, tissue_mask, image_name):
    """
    Affiche l'image avec le masque appliqué et demande à l'utilisateur de valider.
    
    Args:
        image: L'image originale
        tissue_mask: Le masque binaire (True = tissu)
        image_name: Nom de l'image pour l'affichage
    
    Returns:
        Le masque validé (possiblement inversé)
    """
    # Créer l'image masquée
    masked_image = image.copy()
    masked_image[~tissue_mask] = 0
    
    # Afficher l'image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Image originale
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original - {image_name}")
    axes[0].axis('off')
    
    # Image masquée
    axes[1].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Masque Otsu appliqué")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
    # Demander validation
    while True:
        response = input(f"\nLe masque pour '{image_name}' est-il correct? (o=oui, n=non/inverser, q=quitter): ").strip().lower()
        
        if response == 'o':
            plt.close(fig)
            return tissue_mask
        elif response == 'n':
            plt.close(fig)
            # Inverser le masque
            inverted_mask = ~tissue_mask
            print("  ✓ Masque inversé")
            return inverted_mask
        elif response == 'q':
            plt.close(fig)
            print("\nArrêt du traitement demandé par l'utilisateur.")
            exit(0)
        else:
            print("  Réponse invalide. Veuillez entrer 'o', 'n' ou 'q'.")


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


def create_tiles(image_path, output_base_dir, mask_output_dir, tile_size=256, tissue_threshold=0.5, min_tiles=10, morphology_kernel_size=5, gaussian_blur_size=0, skip_existing=True):
    """
    Découpe une image en tiles de taille fixe avec masque Otsu.
    
    Args:
        image_path: Chemin vers l'image WSI
        output_base_dir: Dossier de sortie de base
        mask_output_dir: Dossier de sortie pour les masques Otsu
        tile_size: Taille des tiles (256x256 par défaut)
        tissue_threshold: Seuil de tissu minimum pour garder un tile
        min_tiles: Nombre minimum de tiles pour créer le dossier patient
        morphology_kernel_size: Taille du kernel morphologique (plus grand = comble plus de trous)
        gaussian_blur_size: Taille du flou gaussien avant Otsu (0 = désactivé)
        skip_existing: Si True, ignore les images déjà traitées
    
    Returns:
        Nombre de tiles créés, ou -1 si l'image a été ignorée (déjà traitée)
    """
    # Extraire le nom du fichier pour vérifier si déjà traité
    filename = Path(image_path).name
    image_name = Path(image_path).stem
    
    # Vérifier si l'image a déjà été traitée (masque Otsu existe)
    if skip_existing:
        mask_dir = Path(mask_output_dir)
        mask_filename = f"{image_name}_masked.png"
        mask_path = mask_dir / mask_filename
        
        if mask_path.exists():
            return -1  # Image déjà traitée, on l'ignore
    
    # Charger l'image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Erreur: impossible de charger {image_path}")
        return 0
    
    h, w = image.shape[:2]
    
    # Appliquer le masque d'Otsu
    tissue_mask = apply_otsu_mask(image, morphology_kernel_size, gaussian_blur_size)
    
    # Extraire l'ID patient
    patient_id = extract_patient_id(filename)
    
    # Validation interactive du masque
    tissue_mask = validate_mask_interactive(image, tissue_mask, filename)
    
    # Sauvegarder l'image avec le masque validé (fond noir)
    mask_dir = Path(mask_output_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Appliquer le masque à l'image (fond noir pour les zones sans tissu)
    masked_image = image.copy()
    masked_image[~tissue_mask] = 0  # Mettre en noir les zones sans tissu
    
    mask_filename = f"{image_name}_masked.png"
    mask_path = mask_dir / mask_filename
    cv2.imwrite(str(mask_path), masked_image)
    
    # Découper en tiles sans overlapping et collecter les tiles valides
    valid_tiles = []
    
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            # Extraire le tile
            tile = image[y:y+tile_size, x:x+tile_size]
            mask_tile = tissue_mask[y:y+tile_size, x:x+tile_size]
            
            # Vérifier si le tile contient suffisamment de tissu
            if has_tissue(tile, mask_tile, tissue_threshold):
                valid_tiles.append(tile)
    
    # Ne sauvegarder les tiles que s'il y en a au moins min_tiles
    if len(valid_tiles) >= min_tiles:
        # Créer le dossier de sortie pour ce patient
        patient_dir = Path(output_base_dir) / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder tous les tiles valides
        for idx, tile in enumerate(valid_tiles, start=1):
            tile_filename = f"{image_name}_{idx}.png"
            tile_path = patient_dir / tile_filename
            cv2.imwrite(str(tile_path), tile)
        
        return len(valid_tiles)
    else:
        return 0


def process_dataset(dataset_dir, output_dir, mask_output_dir, tile_size=256, tissue_threshold=0.5, min_tiles=10, morphology_kernel_size=5, gaussian_blur_size=0, skip_existing=True):
    """
    Traite toutes les images du dataset.
    
    Args:
        dataset_dir: Dossier contenant les images WSI
        output_dir: Dossier de sortie pour les tiles
        mask_output_dir: Dossier de sortie pour les masques Otsu
        tile_size: Taille des tiles
        tissue_threshold: Seuil de tissu minimum
        min_tiles: Nombre minimum de tiles pour créer le dossier patient
        morphology_kernel_size: Taille du kernel morphologique (plus grand = comble plus de trous)
        gaussian_blur_size: Taille du flou gaussien avant Otsu (0 = désactivé)
        skip_existing: Si True, ignore les images déjà traitées (reprise automatique)
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
    print(f"Minimum de tiles: {min_tiles}")
    print(f"Flou gaussien: {gaussian_blur_size}x{gaussian_blur_size} (0=désactivé)")
    print(f"Kernel morphologique: {morphology_kernel_size}x{morphology_kernel_size} (0=désactivé)")
    print(f"Reprise automatique: {'activée' if skip_existing else 'désactivée'}")
    print(f"Dossier de sortie tiles: {output_dir}")
    print(f"Dossier de sortie masques: {mask_output_dir}")
    print()
    
    total_tiles = 0
    skipped_images = 0
    
    # Traiter chaque image avec barre de progression
    for image_path in tqdm(images, desc="Traitement des WSI", unit="image"):
        try:
            tiles_created = create_tiles(
                image_path,
                output_path,
                mask_output_dir,
                tile_size=tile_size,
                tissue_threshold=tissue_threshold,
                min_tiles=min_tiles,
                morphology_kernel_size=morphology_kernel_size,
                gaussian_blur_size=gaussian_blur_size,
                skip_existing=skip_existing
            )
            
            if tiles_created == -1:
                # Image déjà traitée, on l'ignore
                skipped_images += 1
                tqdm.write(f"  {image_path.name}: déjà traité, ignoré")
            elif tiles_created > 0:
                total_tiles += tiles_created
                tqdm.write(f"  {image_path.name}: {tiles_created} tiles créés")
            else:
                tqdm.write(f"  {image_path.name}: ignoré (< {min_tiles} tiles)")
        except Exception as e:
            tqdm.write(f"  Erreur avec {image_path.name}: {str(e)}")
    
    print()
    print(f"Terminé! Total: {total_tiles} tiles créés")
    if skip_existing and skipped_images > 0:
        print(f"Images déjà traitées ignorées: {skipped_images}")


if __name__ == "__main__":
    # Configuration
    DATASET_DIR = "dataset"
    OUTPUT_DIR = "dataset_tiles"
    MASK_OUTPUT_DIR = "otsu_mask"
    TILE_SIZE = 256
    TISSUE_THRESHOLD = 0.5  # 50% minimum de tissu dans le tile
    MIN_TILES = 10  # Minimum de tiles pour créer un dossier patient
    GAUSSIAN_BLUR_SIZE = 21  # Flou gaussien avant Otsu (0=désactivé, 11-31 recommandé pour WSI)
    MORPHOLOGY_KERNEL_SIZE = 15  # Taille du kernel pour combler les trous (0=désactivé, 5-20 recommandé)
    
    # Lancer le traitement
    process_dataset(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        mask_output_dir=MASK_OUTPUT_DIR,
        tile_size=TILE_SIZE,
        tissue_threshold=TISSUE_THRESHOLD,
        min_tiles=MIN_TILES,
        morphology_kernel_size=MORPHOLOGY_KERNEL_SIZE,
        gaussian_blur_size=GAUSSIAN_BLUR_SIZE
    )
