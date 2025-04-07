# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run scripts: `python LQN_all_case.py --num_system 3 --num_ancilla 1 --type 0`
- Run parallel execution: `python parallel/LQN_all_case_parallel.py --num_system 3 --num_ancilla 1 --n_workers 16 --type "default"`
- Batch submission: `sbatch --export=VAR1=3,VAR2=1,VAR3=0 run_LQN.sbatch`

## Code Style Guidelines
- **Imports**: Standard library imports first, followed by third-party packages (networkx, igraph, numpy, etc.)
- **Naming**: Use snake_case for functions and variables
- **Function naming**: Use descriptive names with prefixes like `EPM_`, `find_`, `generate_`
- **Documentation**: Add comments for complex logic and function descriptions
- **Graph handling**: Support both networkx and igraph implementations
- **Data structures**: Use Counter and defaultdict for frequency counting
- **Error handling**: Use explicit error checking rather than try/except blocks

## Project Structure
- Main code in Python (.py) files and Jupyter notebooks (.ipynb)
- Parallel implementations in `/parallel` directory
- Use pickle for serialization of quantum states