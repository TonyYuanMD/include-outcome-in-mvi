#!/bin/bash
#SBATCH --job-name=simulation_mvi
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00          # Maximum runtime (adjust based on your config)
#SBATCH --cpus-per-task=64      # Number of CPU cores (prioritize CPU over GPU)
#SBATCH --mem=64G                # Memory per node (adjust based on your needs)
#SBATCH --gres=gpu:0             # No GPU needed (simulation is CPU-bound)
#SBATCH --partition=normal       # Partition name (check your HPC's available partitions)
#SBATCH --mail-type=END,FAIL     # Email notifications (optional)
#SBATCH --mail-user=your.email@example.com  # Your email (optional)

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Load required modules (adjust based on your HPC system)
# Example for systems with module system:
# module load python/3.10
# module load anaconda3

# Or activate conda environment if using conda
# source activate mvi_env
# conda activate mvi_env

# Set Python path (if needed)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Set number of processes based on SLURM allocation
export NUM_PROCESSES=$SLURM_CPUS_PER_TASK

# Configuration file (change this to your desired config)
CONFIG_FILE="${1:-config_full_factorial.json}"

echo "Running simulation with config: $CONFIG_FILE"
echo "Using $NUM_PROCESSES CPU cores"
echo ""

# Run simulation
python -c "
from run_simulation import run_simulation
import os
import sys

# Update num_cores in run_simulation if needed
# This will be picked up by the modified run_simulation.py
os.environ['SLURM_CPUS_PER_TASK'] = str($NUM_PROCESSES)

# Run simulation
try:
    results_all, results_avg = run_simulation(config_file='$CONFIG_FILE')
    print('\\nSimulation completed successfully!')
    print(f'Results saved in results/report/')
except Exception as e:
    print(f'\\nERROR: Simulation failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# Print completion information
echo ""
echo "End Time: $(date)"
echo "Job completed"

# Optional: Compress results to save space
# tar -czf results_${SLURM_JOB_ID}.tar.gz results/

