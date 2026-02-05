<img width="2242" height="1267" alt="image" src="https://github.com/user-attachments/assets/3e0d6e0b-aafa-400b-a129-63b837c2ee01" />

# ClamV2 - Guide d'Utilisation
Ce guide vous aidera à utiliser le projet ClamV2 pour l'extraction de features, l'entraînement de modèles et l'inférence.

```bash
# 1. Extraire les features
python src/extract_features.py --tiles_dir dataset_tiles --output_dir features --tile_size 256

# 2. Entraîner le modèle
python train.py --clinical_csv clinical_data.csv --features_dir features --output_dir results

# 3. Inférence et heatmaps
python infer.py --patient_id 13901 --full_resolution
```

## Création de l'environnement virtuel
```bash
module load python
python3 -m venv venv
source venv/bin/activate
pip3 install --prefix=/work/imvia/in156281/clamv2/venv -r requirements.txt
export PYTHONPATH=/work/imvia/in156281/clamv2/venv/lib/python3.9/site-packages:$PYTHONPATH
pip3 list
```
## Alias 'venv'
```bash
mkdir -p /work/imvia/in156281/.cache/matplotlib
mkdir -p /work/imvia/in156281/.cache/wandb
mkdir -p /work/imvia/in156281/.config/wandb
alias venv='module load python && source venv/bin/activate 
                               && export PYTHONPATH=venv/lib/python3.9/site-packages:$PYTHONPATH
                               && export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib
                               && export WANDB_CACHE_DIR=/work/imvia/in156281/.cache/wandb 
                               && export WANDB_CONFIG_DIR=/work/imvia/in156281/.config/wandb'
```

# Résultats du modèle 
Val Set: val_loss=0.5365, val_error=0.3077, auc=0.6863
  Instance class 0 clustering acc: 0.6763, correct=211/312
  Instance class 1 clustering acc: 0.6955, correct=217/312

Test Results:
  Loss: 0.5365
  Error: 0.3077
  AUC: 0.6863
  Accuracy: 0.6923
