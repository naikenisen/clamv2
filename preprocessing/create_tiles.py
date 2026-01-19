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
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont


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


class MaskValidationGUI:
    """
    Interface Tkinter pour valider les masques de manière interactive.
    Affiche les masques par lots pour économiser les ressources graphiques.
    """
    
    def __init__(self, images_data, preview_max_size=800, images_per_page=20):
        """
        Args:
            images_data: Liste de dictionnaires contenant:
                - 'path': chemin de l'image
                - 'patient_id': ID du patient
                - 'image_name': nom de l'image
                - 'morphology_kernel_size': taille du kernel
                - 'gaussian_blur_size': taille du flou
            preview_max_size: Taille maximale pour charger les images (résolution réduite)
            images_per_page: Nombre d'images à afficher par page
        """
        self.images_data = images_data
        self.preview_max_size = preview_max_size
        self.images_per_page = images_per_page
        self.current_page = 0
        self.total_pages = (len(images_data) + images_per_page - 1) // images_per_page
        self.inverted_indices = set()  # Indices des masques à inverser
        self.should_process = False  # Si True, lancer le traitement
        
        # Créer la fenêtre principale
        self.root = tk.Tk()
        self.root.title(f"Validation des masques - {len(images_data)} images")
        self.root.geometry("1400x900")
        
        # Frame du haut avec instructions et boutons
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        instructions = ttk.Label(
            top_frame,
            text="Cliquez sur les masques à INVERSER (bordure rouge). Cliquez à nouveau pour annuler.",
            font=("Arial", 11, "bold")
        )
        instructions.pack(side=tk.LEFT, padx=10)
        
        # Boutons
        button_frame = ttk.Frame(top_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.validate_btn = ttk.Button(
            button_frame,
            text=f"✓ Valider et créer les tiles",
            command=self.validate_and_process
        )
        self.validate_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ttk.Button(
            button_frame,
            text="✗ Annuler",
            command=self.cancel
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Frame de pagination
        pagination_frame = ttk.Frame(self.root, padding="10")
        pagination_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.prev_btn = ttk.Button(
            pagination_frame,
            text="◀ Page précédente",
            command=self.prev_page,
            state=tk.DISABLED
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.page_label = ttk.Label(
            pagination_frame,
            text=f"Page {self.current_page + 1} / {self.total_pages}",
            font=("Arial", 10, "bold")
        )
        self.page_label.pack(side=tk.LEFT, padx=20)
        
        self.next_btn = ttk.Button(
            pagination_frame,
            text="Page suivante ▶",
            command=self.next_page
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # Label pour afficher le nombre d'inversions
        self.invert_count_label = ttk.Label(
            pagination_frame,
            text="0 masques à inverser",
            font=("Arial", 10)
        )
        self.invert_count_label.pack(side=tk.RIGHT, padx=20)
        
        # Frame avec canvas et scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas avec scrollbar verticale
        self.canvas = tk.Canvas(main_frame, bg="white")
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind pour redimensionner le frame interne
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mousewheel pour scroll
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)  # Linux scroll up
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)  # Linux scroll down
        
        # Afficher la première page
        self.display_current_page()
        
    def _on_canvas_configure(self, event):
        """Ajuste la largeur du frame interne à celle du canvas"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _on_mousewheel(self, event):
        """Gère le scroll avec la molette"""
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
    
    def prev_page(self):
        """Affiche la page précédente"""
        if self.current_page > 0:
            self.current_page -= 1
            self.display_current_page()
    
    def next_page(self):
        """Affiche la page suivante"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.display_current_page()
    
    def display_current_page(self):
        """Affiche les images de la page courante"""
        # Nettoyer le frame scrollable
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Calculer les indices de début et fin
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, len(self.images_data))
        
        # Créer la grille pour cette page
        self.create_image_grid(start_idx, end_idx)
        
        # Mettre à jour les boutons
        self.prev_btn.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_page < self.total_pages - 1 else tk.DISABLED)
        self.page_label.config(text=f"Page {self.current_page + 1} / {self.total_pages}")
        
        # Reset scroll
        self.canvas.yview_moveto(0)
    
    def create_image_grid(self, start_idx, end_idx):
        """Crée la grille d'images avec masques"""
        # Calculer le nombre de colonnes (3 colonnes de paires original/masque)
        pairs_per_row = 3
        
        # Créer les miniatures
        self.photo_images = []  # Garder les références
        
        # Obtenir les images pour cette page
        page_images = self.images_data[start_idx:end_idx]
        
        for relative_idx, data in enumerate(page_images):
            idx = start_idx + relative_idx  # Index global
            row = relative_idx // pairs_per_row
            col = (relative_idx % pairs_per_row) * 2  # *2 car chaque paire prend 2 colonnes
            
            # Charger et redimensionner l'image à la volée
            image = cv2.imread(data['path'])
            if image is None:
                continue
            
            # Redimensionner l'image pour économiser la mémoire
            h, w = image.shape[:2]
            if max(h, w) > self.preview_max_size:
                scale = self.preview_max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Générer le masque sur l'image redimensionnée
            tissue_mask = apply_otsu_mask(
                image,
                data['morphology_kernel_size'],
                data['gaussian_blur_size']
            )
            
            # Frame pour cette paire d'images
            pair_frame = ttk.Frame(self.scrollable_frame, relief=tk.RAISED, borderwidth=2)
            pair_frame.grid(row=row, column=col, columnspan=2, padx=5, pady=5, sticky="nsew")
            
            # Label avec le nom de l'image
            name_label = ttk.Label(
                pair_frame,
                text=f"{data['image_name']} (Patient: {data['patient_id']})",
                font=("Arial", 9, "bold")
            )
            name_label.grid(row=0, column=0, columnspan=2, pady=(0, 5))
            
            # Image originale
            img_original = self.create_thumbnail(image, 250)
            photo_original = ImageTk.PhotoImage(img_original)
            self.photo_images.append(photo_original)
            
            label_original = tk.Label(pair_frame, image=photo_original)
            label_original.grid(row=1, column=0, padx=5, pady=5)
            
            title_original = ttk.Label(pair_frame, text="Original", font=("Arial", 8))
            title_original.grid(row=2, column=0)
            
            # Image avec masque (clickable)
            masked_img = image.copy()
            masked_img[~tissue_mask] = 0
            img_masked = self.create_thumbnail(masked_img, 250)
            photo_masked = ImageTk.PhotoImage(img_masked)
            self.photo_images.append(photo_masked)
            
            # Frame pour le masque avec bordure colorée
            # Vérifier si ce masque est dans la liste des inversions
            border_color = "red" if idx in self.inverted_indices else "green"
            border_width = 5 if idx in self.inverted_indices else 3
            mask_frame = tk.Frame(pair_frame, relief=tk.SOLID, borderwidth=border_width, bg=border_color)
            mask_frame.grid(row=1, column=1, padx=5, pady=5)
            
            label_masked = tk.Label(mask_frame, image=photo_masked, cursor="hand2")
            label_masked.pack()
            
            # Bind click event
            label_masked.bind("<Button-1>", lambda e, i=idx, f=mask_frame: self.toggle_invert(i, f))
            
            title_masked = ttk.Label(pair_frame, text="Masque (cliquer pour inverser)", font=("Arial", 8))
            title_masked.grid(row=2, column=1)
            
            # Stocker la référence au frame pour changer la bordure
            data['mask_frame'] = mask_frame
            
            # Libérer la mémoire
            del image, tissue_mask, masked_img
    
    def create_thumbnail(self, image_bgr, max_size):
        """Crée une miniature PIL de l'image"""
        # Convertir BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Créer image PIL
        pil_img = Image.fromarray(image_rgb)
        
        # Redimensionner en gardant le ratio
        pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return pil_img
    
    def toggle_invert(self, idx, frame):
        """Toggle l'inversion pour une image"""
        if idx in self.inverted_indices:
            # Retirer de la liste
            self.inverted_indices.remove(idx)
            frame.config(bg="green", borderwidth=3)
        else:
            # Ajouter à la liste
            self.inverted_indices.add(idx)
            frame.config(bg="red", borderwidth=5)
        
        # Mettre à jour les labels
        count = len(self.inverted_indices)
        self.validate_btn.config(
            text=f"✓ Valider et créer les tiles ({count} masques à inverser)" if count > 0 
                 else "✓ Valider et créer les tiles"
        )
        self.invert_count_label.config(
            text=f"{count} masque{'s' if count > 1 else ''} à inverser"
        )
    
    def validate_and_process(self):
        """Valide les choix et lance le traitement"""
        if self.inverted_indices:
            response = messagebox.askyesno(
                "Confirmation",
                f"Vous allez inverser {len(self.inverted_indices)} masque(s).\n\n"
                f"Continuer avec la création des tiles ?"
            )
        else:
            response = messagebox.askyesno(
                "Confirmation",
                f"Aucun masque à inverser.\n\n"
                f"Continuer avec la création des tiles ?"
            )
        
        if response:
            self.should_process = True
            self.root.quit()
            self.root.destroy()
    
    def cancel(self):
        """Annule le traitement"""
        response = messagebox.askyesno(
            "Annuler",
            "Êtes-vous sûr de vouloir annuler ?"
        )
        if response:
            self.should_process = False
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        """Lance la GUI et retourne les choix"""
        self.root.mainloop()
        return self.should_process, self.inverted_indices


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


def create_tiles(image_path, mask_output_dir, tile_size=256, tissue_threshold=0.5, morphology_kernel_size=5, gaussian_blur_size=0, invert_mask=False):
    """
    Découpe une image en tiles de taille fixe avec masque Otsu.
    
    Args:
        image_path: Chemin vers l'image WSI
        mask_output_dir: Dossier de sortie pour les masques Otsu
        tile_size: Taille des tiles (256x256 par défaut)
        tissue_threshold: Seuil de tissu minimum pour garder un tile
        morphology_kernel_size: Taille du kernel morphologique (plus grand = comble plus de trous)
        gaussian_blur_size: Taille du flou gaussien avant Otsu (0 = désactivé)
        invert_mask: Si True, inverse le masque
    
    Returns:
        Dictionnaire avec patient_id, image_name, tiles et nombre de tiles créés
    """
    # Extraire le nom du fichier
    filename = Path(image_path).name
    image_name = Path(image_path).stem
    
    # Charger l'image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Erreur: impossible de charger {image_path}")
        return {'patient_id': None, 'image_name': image_name, 'tiles': [], 'count': 0}
    
    h, w = image.shape[:2]
    
    # Appliquer le masque d'Otsu
    tissue_mask = apply_otsu_mask(image, morphology_kernel_size, gaussian_blur_size)
    
    # Inverser si demandé
    if invert_mask:
        tissue_mask = ~tissue_mask
    
    # Extraire l'ID patient
    patient_id = extract_patient_id(filename)
    
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
                valid_tiles.append((tile, y, x))
    
    # Retourner les informations
    return {
        'patient_id': patient_id,
        'image_name': image_name,
        'tiles': valid_tiles,
        'count': len(valid_tiles)
    }


def process_dataset(dataset_dir, output_dir, mask_output_dir, tile_size=256, tissue_threshold=0.5, min_tiles=10, morphology_kernel_size=5, gaussian_blur_size=0, images_per_page=15):
    """
    Traite toutes les images du dataset avec interface GUI.
    Pour chaque patient, ne garde que les tiles de la lame la plus riche.
    
    Args:
        dataset_dir: Dossier contenant les images WSI
        output_dir: Dossier de sortie pour les tiles
        mask_output_dir: Dossier de sortie pour les masques Otsu
        tile_size: Taille des tiles
        tissue_threshold: Seuil de tissu minimum
        min_tiles: Nombre minimum de tiles pour créer le dossier patient
        morphology_kernel_size: Taille du kernel morphologique
        gaussian_blur_size: Taille du flou gaussien avant Otsu
        images_per_page: Nombre d'images par page dans l'interface GUI
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
    
    print(f"=" * 80)
    print(f"PHASE 1: Chargement et préparation des masques")
    print(f"=" * 80)
    print(f"Images trouvées: {len(images)}")
    print(f"Taille des tiles: {tile_size}x{tile_size} pixels")
    print(f"Seuil de tissu: {tissue_threshold * 100}%")
    print(f"Minimum de tiles: {min_tiles}")
    print(f"Flou gaussien: {gaussian_blur_size}x{gaussian_blur_size} (0=désactivé)")
    print(f"Kernel morphologique: {morphology_kernel_size}x{morphology_kernel_size} (0=désactivé)")
    print()
    
    # Préparer les données pour la GUI (sans charger les images en mémoire)
    images_data = []
    
    for image_path in tqdm(images, desc="Préparation des métadonnées", unit="image"):
        try:
            # Vérifier que l'image existe et est lisible
            if not image_path.exists():
                tqdm.write(f"  Erreur: fichier introuvable {image_path.name}")
                continue
            
            # Extraire les infos sans charger l'image
            filename = image_path.name
            image_name = Path(image_path).stem
            patient_id = extract_patient_id(filename)
            
            images_data.append({
                'path': str(image_path),
                'patient_id': patient_id,
                'image_name': image_name,
                'morphology_kernel_size': morphology_kernel_size,
                'gaussian_blur_size': gaussian_blur_size
            })
            
        except Exception as e:
            tqdm.write(f"  Erreur avec {image_path.name}: {str(e)}")
    
    if not images_data:
        print("Aucune image n'a pu être trouvée.")
        return
    
    print()
    print(f"=" * 80)
    print(f"PHASE 2: Validation interactive des masques")
    print(f"=" * 80)
    print(f"Ouverture de l'interface graphique...")
    print(f"Affichage par lots de {images_per_page} images")
    print()
    
    # Lancer l'interface GUI
    gui = MaskValidationGUI(images_data, images_per_page=images_per_page)
    should_process, inverted_indices = gui.run()
    
    if not should_process:
        print("\nTraitement annulé par l'utilisateur.")
        return
    
    print()
    print(f"=" * 80)
    print(f"PHASE 3: Création des tiles")
    print(f"=" * 80)
    print(f"Masques à inverser: {len(inverted_indices)}")
    print()
    
    # Dictionnaire pour stocker les résultats par patient
    patient_slides = {}
    
    # Traiter chaque image (une par une pour économiser la mémoire)
    for idx, data in enumerate(tqdm(images_data, desc="Création des tiles", unit="image")):
        try:
            # Vérifier si le masque doit être inversé
            invert_mask = idx in inverted_indices
            
            result = create_tiles(
                data['path'],
                mask_output_dir,
                tile_size=tile_size,
                tissue_threshold=tissue_threshold,
                morphology_kernel_size=data['morphology_kernel_size'],
                gaussian_blur_size=data['gaussian_blur_size'],
                invert_mask=invert_mask
            )
            
            if result['count'] >= min_tiles and result['patient_id'] is not None:
                patient_id = result['patient_id']
                if patient_id not in patient_slides:
                    patient_slides[patient_id] = []
                
                patient_slides[patient_id].append(result)
                
                inv_marker = " [INVERSÉ]" if invert_mask else ""
                tqdm.write(f"  {data['image_name']}: {result['count']} tiles (patient {patient_id}){inv_marker}")
            else:
                tqdm.write(f"  {data['image_name']}: ignoré (< {min_tiles} tiles)")
                
        except Exception as e:
            tqdm.write(f"  Erreur avec {data['image_name']}: {str(e)}")
    
    print()
    print("=" * 80)
    print("PHASE 4: Sélection de la meilleure lame par patient")
    print("=" * 80)
    
    total_tiles = 0
    
    # Pour chaque patient, garder la lame avec le plus de tiles
    for patient_id, slides in patient_slides.items():
        # Trouver la lame avec le plus de tiles
        best_slide = max(slides, key=lambda x: x['count'])
        
        # Afficher les informations
        if len(slides) > 1:
            print(f"\nPatient {patient_id}: {len(slides)} lames trouvées")
            for slide in slides:
                marker = "→ SÉLECTIONNÉE" if slide == best_slide else "  ignorée"
                print(f"  - {slide['image_name']}: {slide['count']} tiles {marker}")
        else:
            print(f"\nPatient {patient_id}: 1 lame ({best_slide['image_name']}, {best_slide['count']} tiles)")
        
        # Créer le dossier de sortie pour ce patient
        patient_dir = output_path / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les tiles de la meilleure lame avec coordonnées
        for tile, y, x in best_slide['tiles']:
            tile_filename = f"{y}_{x}.png"
            tile_path = patient_dir / tile_filename
            cv2.imwrite(str(tile_path), tile)
        
        total_tiles += best_slide['count']
    
    print()
    print("=" * 80)
    print("PHASE 5: Nettoyage et harmonisation du dossier dataset")
    print("=" * 80)
    
    # Collecter les images sélectionnées (best slides)
    selected_images = set()
    rename_map = {}  # Ancien nom -> nouveau nom
    
    for patient_id, slides in patient_slides.items():
        best_slide = max(slides, key=lambda x: x['count'])
        # Récupérer le chemin de l'image sélectionnée
        for data in images_data:
            if data['image_name'] == best_slide['image_name']:
                selected_images.add(data['path'])
                
                # Préparer le nouveau nom harmonisé (sans _2, _3, etc.)
                old_path = Path(data['path'])
                extension = old_path.suffix
                new_filename = f"{patient_id}{extension}"
                new_path = old_path.parent / new_filename
                
                # Ajouter au mapping uniquement si le nom change
                if old_path.name != new_filename:
                    rename_map[str(old_path)] = str(new_path)
                
                break
    
    # Supprimer les images non sélectionnées
    deleted_count = 0
    for image_path in images:
        if str(image_path) not in selected_images:
            try:
                image_path.unlink()
                deleted_count += 1
                print(f"  ✗ Supprimé: {image_path.name}")
            except Exception as e:
                print(f"  ⚠ Erreur lors de la suppression de {image_path.name}: {str(e)}")
    
    print(f"\n{deleted_count} image(s) non sélectionnée(s) supprimée(s)")
    
    # Renommer les images sélectionnées pour harmoniser les noms
    renamed_count = 0
    for old_path, new_path in rename_map.items():
        try:
            Path(old_path).rename(new_path)
            renamed_count += 1
            print(f"  ➜ Renommé: {Path(old_path).name} → {Path(new_path).name}")
        except Exception as e:
            print(f"  ⚠ Erreur lors du renommage de {Path(old_path).name}: {str(e)}")
    
    if renamed_count == 0:
        print("\nAucun renommage nécessaire (noms déjà harmonisés)")
    else:
        print(f"\n{renamed_count} image(s) renommée(s) pour harmonisation")
    
    print()
    print("=" * 80)
    print(f"✓ TERMINÉ!")
    print(f"Total: {total_tiles} tiles sauvegardés pour {len(patient_slides)} patients")
    print(f"Dataset nettoyé: {len(selected_images)} WSI sélectionnées conservées")
    print(f"Dossier de sortie des tiles: {output_dir}")
    print(f"Dossier dataset: {dataset_dir}")
    print("=" * 80)


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
    IMAGES_PER_PAGE = 15  # Nombre d'images par page dans l'interface
    
    # Lancer le traitement
    process_dataset(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        mask_output_dir=MASK_OUTPUT_DIR,
        tile_size=TILE_SIZE,
        tissue_threshold=TISSUE_THRESHOLD,
        min_tiles=MIN_TILES,
        morphology_kernel_size=MORPHOLOGY_KERNEL_SIZE,
        gaussian_blur_size=GAUSSIAN_BLUR_SIZE,
        images_per_page=IMAGES_PER_PAGE
    )
