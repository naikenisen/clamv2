# ClamV2 - Guide d'Utilisation
Ce guide vous aidera à utiliser le projet ClamV2 pour l'extraction de features, l'entraînement de modèles et l'inférence.

```bash
# 1. Extraire les features
python src/extract_features.py --tiles_dir dataset_tiles --output_dir features

# 2. Entraîner le modèle
python train.py --clinical_csv clinical_data.csv --features_dir features --output_dir results

# 3. Inférence et heatmaps
python infer.py --results_dir results --features_dir features --output_dir inference_output
```