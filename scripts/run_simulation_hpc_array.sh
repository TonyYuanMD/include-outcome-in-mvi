#!/bin/bash
#SBATCH --job-name=simulation_array
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0-107          # For 108 combinations (0-indexed)
#SBATCH --partition=normal

# Job array script for running parameter combinations in parallel
# This splits the full factorial into separate array jobs

echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Load modules (adjust for your HPC)
# module load python/3.10

# Activate environment
# conda activate mvi_env

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration file
CONFIG_FILE="${1:-config_large_scale.json}"

# Run single combination (you'd need to modify run_simulation to support this)
# For now, this is a template - you'd need to implement combination selection
python -c "
from run_simulation import run_simulation
import json
from itertools import product

# Load config
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

# Get all combinations
combinations = list(product(
    config['n'], config['p'], [config['num_runs']],
    config['continuous_pct'], config['integer_pct'], config['sparsity'],
    config['include_interactions'], config['include_nonlinear'], config['include_splines']
))

# Select combination for this array task
task_id = int('$SLURM_ARRAY_TASK_ID')
if task_id < len(combinations):
    # Run single combination (would need to implement this)
    print(f'Running combination {task_id} of {len(combinations)}')
    # run_single_combination(...)
else:
    print(f'Task ID {task_id} exceeds number of combinations')
"

echo ""
echo "End Time: $(date)"

