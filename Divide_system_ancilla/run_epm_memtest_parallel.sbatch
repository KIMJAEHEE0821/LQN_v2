#!/bin/bash
#SBATCH -J 4_2_parallel  # Job name (예: 병렬 실행 명시)
#SBATCH -p wbatch       # 사용할 파티션
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36   # 사용할 코어 수 (★★★★★ 예시 값, 실제 사용 가능/원하는 코어 수로 변경)
#SBATCH --mem-per-cpu=8G     # 코어당 메모리 요청 (★★★★★ 중요: 실제 측정된 MaxRSS 기반으로 수정하세요! 예시 값 8GB)
#SBATCH --output=run_4_2_parallel.out # 출력 파일 이름 변경
#SBATCH --error=run_4_2_parallel.err  # 에러 파일 이름 변경
#SBATCH --time=9999:00:00       # 최대 실행 시간

echo "Start time: $(date)"
ulimit -s unlimited
echo "SLURM job directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR || exit
# SLURM_NPROCS 환경 변수는 요청된 총 코어 수를 나타냅니다 (Python 코드에서 사용됨)
echo "Number of tasks requested (SLURM_NPROCS): $SLURM_NPROCS"
echo "Memory per CPU requested: ${SLURM_MEM_PER_CPU} MB" # SLURM_MEM_PER_CPU는 MB 단위

# Conda 환경 활성화 (경로 확인)
source /home/jaeheekim/miniconda3/etc/profile.d/conda.sh
conda activate lqn
echo "Activated Conda environment: lqn"

# 사용할 파라미터 (예: 4, 2, 0) - 외부에서 export로 전달해도 됨
VAR1=4
VAR2=2
VAR3=0
echo "num_system (VAR1): $VAR1"
echo "num_ancilla (VAR2): $VAR2"
echo "type (VAR3): $VAR3"

# Python 스크립트 직접 실행 (병렬 처리 코드)
echo "Running Python script (main_script.py) with parallel processing..."
# 파이프라인 내에서 SLURM_NPROCS를 읽어 사용하므로 별도 전달 필요 없음
python main_script.py --num_system $VAR1 --num_ancilla $VAR2 --type $VAR3
echo "Python script finished."

echo "End time: $(date)"
