#!/bin/bash
#SBATCH -J 3_3      # Job name
#SBATCH -p wbatch     # batch:28 | sbatch:36 | wbatch:96
#SBATCH --nodes=1     
#SBATCH --ntasks-per-node=1   # Number of cores
#SBATCH --output=run.out     # Standard output
#SBATCH --error=run.err      # Standard error
#SBATCH --time=9999:00:00       # Adjusted time, ensure this matches your job's needs

echo "Start time: $(date)"
s
ulimit -s unlimited

echo "SLURM job directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR || exit

echo "Number of slots: $SLURM_NPROCS"

# Source the Conda environment setup script
source /home/jaeheekim/miniconda3/etc/profile.d/conda.sh

# Activate the specific Conda environment
conda activate lqn


echo "num_system: $VAR1"
echo "num_ancilla: $VAR2"
echo "type: $VAR3"

python LQN_all_case.py --num_system $VAR1 --num_ancilla $VAR2 --type $VAR3

echo "End time: $(date)"