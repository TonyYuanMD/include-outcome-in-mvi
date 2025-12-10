#!/bin/bash
#SBATCH --job-name=simulation_mvi
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err
#SBATCH --time=01:00:00          # Maximum runtime (adjust based on your config)
#SBATCH --cpus-per-task=32       # Number of CPU cores (parallelizes across parameter combinations)
#SBATCH --mem=32G                 # Memory per node (adjust based on CPU count)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Single task (multiprocessing handles parallelism)
#SBATCH --account=ling702w25_class
#SBATCH --gres=gpu:1              # GPU for GAIN/AE (beneficial for larger datasets n>500, p>10)
#SBATCH --partition=gpu           # Partition name (check your HPC's available partitions)
# Note: Parallelization is ACROSS parameter combinations, not runs.
#       Runs are sequential within each combination. For 500 runs, each combination
#       will take time, but multiple combinations run in parallel.
#SBATCH --mail-type=BEGIN,END,FAIL

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
source ~/.bashrc
conda activate CSE595

cd /home/yhongda/include_y_mvi/include-outcome-in-mvi

# Set Python path (if needed)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Set number of processes based on SLURM allocation
# Parallelization is ACROSS parameter combinations (not runs)
# Each process handles one parameter combination and runs all runs sequentially
NUM_PROCESSES=${SLURM_CPUS_PER_TASK:-64}
export NUM_PROCESSES

# Configuration file (change this to your desired config)
CONFIG_FILE="${1:-config_full_factorial.json}"

echo "Running simulation with config: $CONFIG_FILE"
echo "Using $NUM_PROCESSES parallel processes (parallelizing across runs)"
echo "GPU available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "Parallelization strategy:"
echo "  - Single parameter combination detected"
echo "  - Runs are parallelized across $NUM_PROCESSES CPUs"
echo "  - For 100 runs with $NUM_PROCESSES CPUs: ~$(( (100 + NUM_PROCESSES - 1) / NUM_PROCESSES )) batches"
echo ""

# Start CPU monitoring in background (logs to file)
MONITOR_LOG="log/cpu_monitor_${SLURM_JOB_ID}.log"
mkdir -p log
echo "Starting CPU monitoring (logging to $MONITOR_LOG)"
(
    echo "Timestamp,CPU_Usage(%),Idle_Cores,Active_Cores,Total_Cores,Python_Processes"
    while true; do
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        # Get CPU usage percentage using multiple methods for reliability
        # Method 1: Using top (most reliable)
        if command -v top &> /dev/null; then
            CPU_IDLE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | head -1)
            CPU_USAGE=$(echo "$CPU_IDLE" | awk '{printf "%.1f", 100 - $1}')
        # Method 2: Using /proc/stat (fallback)
        elif [ -f /proc/stat ]; then
            CPU_LINE=$(grep "^cpu " /proc/stat)
            if [ ! -z "$CPU_LINE" ]; then
                # Calculate CPU usage from /proc/stat
                CPU_USAGE=$(echo "$CPU_LINE" | awk '{idle=$5+$6; total=$2+$3+$4+$5+$6+$7+$8; if(total>0) printf "%.1f", 100*(1-idle/total); else print "0"}')
            else
                CPU_USAGE="N/A"
            fi
        else
            CPU_USAGE="N/A"
        fi
        
        # Count active Python processes
        ACTIVE_PROCS=$(ps aux | grep -E "[p]ython.*run_simulation|[p]ython -c.*run_simulation" | wc -l)
        
        # Get number of CPU cores
        TOTAL_CORES=${SLURM_CPUS_PER_TASK:-$(nproc)}
        
        # Calculate idle/active cores based on CPU usage
        if [ "$CPU_USAGE" != "N/A" ] && [ ! -z "$CPU_USAGE" ]; then
            IDLE_CORES=$(echo "$TOTAL_CORES $CPU_USAGE" | awk '{printf "%.1f", $1 * (1 - $2/100)}')
            ACTIVE_CORES=$(echo "$TOTAL_CORES $CPU_USAGE" | awk '{printf "%.1f", $1 * ($2/100)}')
        else
            IDLE_CORES="N/A"
            ACTIVE_CORES="N/A"
        fi
        
        echo "$TIMESTAMP,$CPU_USAGE,$IDLE_CORES,$ACTIVE_CORES,$TOTAL_CORES,$ACTIVE_PROCS"
        sleep 60  # Update every 10 seconds
    done
) > "$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
echo "CPU monitor PID: $MONITOR_PID (logging to $MONITOR_LOG)"
echo "To view in real-time: tail -f $MONITOR_LOG"
echo ""

# Run simulation
python -c "
from run_simulation import run_simulation
import os
import sys

# SLURM_CPUS_PER_TASK is automatically set by SLURM and will be picked up by run_simulation.py
# No need to set it manually here

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

# Stop CPU monitoring
if [ ! -z "$MONITOR_PID" ]; then
    echo "Stopping CPU monitoring (PID: $MONITOR_PID)"
    kill $MONITOR_PID 2>/dev/null
    wait $MONITOR_PID 2>/dev/null
    echo "CPU monitoring stopped. Summary saved to $MONITOR_LOG"
    echo ""
    echo "=== CPU Usage Summary ==="
    if [ -f "$MONITOR_LOG" ]; then
        echo "Average CPU Usage: $(tail -n +2 "$MONITOR_LOG" | awk -F',' '{sum+=$2; count++} END {if(count>0) printf "%.1f%%", sum/count; else print "N/A"}')"
        echo "Min CPU Usage: $(tail -n +2 "$MONITOR_LOG" | awk -F',' '{if(NR==1 || $2<min) min=$2} END {printf "%.1f%%", min}')"
        echo "Max CPU Usage: $(tail -n +2 "$MONITOR_LOG" | awk -F',' '{if($2>max) max=$2} END {printf "%.1f%%", max}')"
        echo "Average Idle Cores: $(tail -n +2 "$MONITOR_LOG" | awk -F',' '{sum+=$3; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}')"
    fi
    echo "========================="
    echo ""
fi

# Print completion information
echo ""
echo "End Time: $(date)"
echo "Job completed"

# Optional: Compress results to save space
# tar -czf results_${SLURM_JOB_ID}.tar.gz results/

