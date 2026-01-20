#!/bin/ksh 
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N clamv2v2
cd $WORKDIR
cd /beegfs/data/work/imvia/in156281/clamv2
source /beegfs/data/work/imvia/in156281/clamv2/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/clamv2/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib
python /beegfs/data/work/imvia/in156281/clamv2/train.py\
    --clinical_csv clinical_data.csv \
    --features_dir features \
    --output_dir results \
    --model_type clam_sb \
    --model_size small \
    --dropout 0.5 \
    --lr 0.0001 \
    --bag_weight 0.7 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --es_mode loss \
    --patience 20 \
    --max_grad_norm 1.0 \
    --k_sample 12 \
    --use_class_weights \
    --label_smoothing 0.1 \
    --warmup_epochs 5 \
    --max_epochs 150