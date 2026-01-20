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