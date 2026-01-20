#!/bin/ksh 
#$ -q gpu
#$ -o result_gridsearch.out
#$ -j y
#$ -N clam_gridsearch
cd $WORKDIR
cd /beegfs/data/work/imvia/in156281/clamv2
source /beegfs/data/work/imvia/in156281/clamv2/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/clamv2/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib

python /beegfs/data/work/imvia/in156281/clamv2/gridsearch_cv.py \
    --clinical_csv clinical_data.csv \
    --features_dir features \
    --output_dir results_gridsearch \
    --n_folds 5 \
    --bag_weights 0.5,0.6,0.7,0.8,0.9 \
    --dropouts 0.4,0.5,0.6 \
    --k_samples 6,8,12 \
    --model_type clam_sb \
    --model_size small \
    --embed_dim 768 \
    --lr 0.0001 \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --use_focal_loss \
    --focal_gamma 2.0 \
    --label_smoothing 0.1 \
    --bag_dropout 0.15 \
    --warmup_epochs 5 \
    --max_epochs 100 \
    --patience 15
