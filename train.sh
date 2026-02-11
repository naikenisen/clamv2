#!/bin/ksh 
#$ -q gpu
#$ -j y
#$ -N clam_gridsearch
cd $WORKDIR
cd /beegfs/data/work/imvia/in156281/clamv2
source /beegfs/data/work/imvia/in156281/clamv2/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/clamv2/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib

# Create date-based output directory
DATE_DIR=$(date +%Y-%m-%d)
OUTPUT_DIR="results_${DATE_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Redirect all output to the dated folder
exec > "${OUTPUT_DIR}/result_gridsearch.out" 2>&1

python /beegfs/data/work/imvia/in156281/clamv2/train.py