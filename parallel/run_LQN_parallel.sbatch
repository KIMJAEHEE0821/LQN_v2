#!/bin/bash
#SBATCH -J epm_process  # Job name
#SBATCH -p wbatch  # batch:28 | sbatch:36 | wbatch:96
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # Number of tasks per node
#SBATCH --cpus-per-task=16   # Number of CPU cores per task (adjust according to your server environment)
#SBATCH --mem=64G            # Memory allocation per task (adjust as needed)
#SBATCH --output=epm_%j.out  # Standard output (%j is replaced with job ID)
#SBATCH --error=epm_%j.err   # Standard error
#SBATCH --time=72:00:00      # Maximum execution time (3 days, adjust as needed)
#SBATCH -w wnode02

# Variables are passed when submitting the job: sbatch --export=VAR1=3,VAR2=3 run_job.sh
# VAR1: num_system, VAR2: num_ancilla

echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node name: $SLURMD_NODENAME"
echo "Job directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR || exit

# Print system information
echo "Allocated CPU cores: $SLURM_CPUS_PER_TASK"
echo "Allocated memory: $SLURM_MEM_PER_NODE MB"

# Remove stack size limit
ulimit -s unlimited
echo "Stack size limit: $(ulimit -s)"

# Setup Conda environment
source /home/jaeheekim/miniconda3/etc/profile.d/conda.sh
conda activate lqn
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Print input parameters
echo "num_system: $VAR1"
echo "num_ancilla: $VAR2"

# Background job for memory usage monitoring
(
  while true; do
    mem_usage=$(free -m | grep Mem | awk '{print $3}')
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Memory usage: ${mem_usage}MB" >> memory_usage_$SLURM_JOB_ID.log
    sleep 60
  done
) &
MONITOR_PID=$!

# Execute EPM process
# The --n_workers parameter is matched to the number of CPUs allocated by SLURM
echo "Executing EPM process..."
python LQN_all_case_parallel.py --num_system $VAR1 --num_ancilla $VAR2 --n_workers $SLURM_CPUS_PER_TASK --type "default"

# Terminate memory monitoring
kill $MONITOR_PID

echo "End time: $(date)"