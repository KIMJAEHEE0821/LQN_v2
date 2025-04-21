#!/bin/bash
#SBATCH -J 4_2_memtest  # Job name (예: 메모리 테스트 명시)
#SBATCH -p wbatch     # 사용할 파티션 (batch:28 | sbatch:36 | wbatch:96)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # 사용할 코어 수 (프로파일링 단계에서는 1개 유지)
#SBATCH --mem=1400G   
#SBATCH --output=run_4_2_memtest.out # 출력 파일 이름 변경
#SBATCH --error=run_4_2_memtest.err  # 에러 파일 이름 변경
#SBATCH --time=9999:00:00       # 최대 실행 시간

echo "Start time: $(date)"

# 스택 제한 해제 (필요한 경우)
ulimit -s unlimited

echo "SLURM job directory: $SLURM_SUBMIT_DIR"
# 작업 제출 디렉토리로 이동 (실패 시 종료)
cd $SLURM_SUBMIT_DIR || exit

echo "Number of slots requested: $SLURM_NPROCS" # SLURM_NPROCS는 총 요청 코어 수

# Conda 환경 설정 스크립트 로드 (경로는 사용자 환경에 맞게 확인)
source /home/jaeheekim/miniconda3/etc/profile.d/conda.sh

# 사용할 Conda 환경 활성화
conda activate lqn
echo "Activated Conda environment: lqn"

# 전달받은 환경 변수 출력 (디버깅용)
echo "num_system (VAR1): $VAR1"
echo "num_ancilla (VAR2): $VAR2"
echo "type (VAR3): $VAR3"

# Python 스크립트 실행 (cProfile 사용, 파일 이름 수정됨)
echo "Running Python script (main_script.py) with cProfile..."
# 결과를 .prof 파일로 저장 (파일 이름에 파라미터 포함)
# LQN_all_case.py -> main_script.py 로 수정됨
python -m cProfile -o profile_sys${VAR1}_anc${VAR2}_type${VAR3}.prof main_script.py --num_system $VAR1 --num_ancilla $VAR2 --type $VAR3
echo "Profiling finished. Output saved to profile_sys${VAR1}_anc${VAR2}_type${VAR3}.prof"

echo "End time: $(date)"