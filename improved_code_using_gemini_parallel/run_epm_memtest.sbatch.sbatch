#!/bin/bash
#SBATCH -J 4_2_memtest  # Job name 
#SBATCH -p wbatch 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --mem=1400G       # memory require 
#SBATCH --output=run_4_2_memtest.out # output file
#SBATCH --error=run_4_2_memtest.err  # erro file
#SBATCH --time=9999:00:00       # runtime

echo "Start time: $(date)"
ulimit -s unlimited
echo "SLURM job directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR || exit
echo "Number of slots requested: $SLURM_NPROCS"

# Conda 환경 활성화 (경로 확인)
source /home/jaeheekim/miniconda3/etc/profile.d/conda.sh
conda activate lqn
echo "Activated Conda environment: lqn"

# 전달받을 환경 변수 (num_system=4, num_ancilla=2, type=0)
VAR1=4
VAR2=2
VAR3=0
echo "num_system (VAR1): $VAR1"
echo "num_ancilla (VAR2): $VAR2"
echo "type (VAR3): $VAR3"

# Python 스크립트 직접 실행 (cProfile 제거됨)
echo "Running Python script (main_script.py)..."
python main_script.py --num_system $VAR1 --num_ancilla $VAR2 --type $VAR3
echo "Python script finished."

echo "End time: $(date)"