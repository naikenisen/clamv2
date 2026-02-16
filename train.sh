#!/bin/ksh 
#$ -q gpu
#$ -j y
#$ -o result.out
#$ -N clamv2
cd $WORKDIR
cd /beegfs/data/work/imvia/in156281/clamv2
source /beegfs/data/work/imvia/in156281/clamv2/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/clamv2/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib

python /beegfs/data/work/imvia/in156281/clamv2/train.py