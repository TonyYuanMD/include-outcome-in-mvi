#!/bin/bash
#SBATCH --job-name=simulation_gpu
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err
#SBATCH --time=05:00:00          # Longer time for GPU methods (GAIN/Autoencoder are slow)
#SBATCH --cpus-per-task=8        # Fewer CPUs to reduce GPU contention
#SBATCH --mem=32G                 # Memory per node
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Single task (multiprocessing handles parallelism)
#SBATCH --account=ling702w25_class
#SBATCH --gres=gpu:1             # GPU for GAIN/Autoencoder
#SBATCH --partition=gpu           # GPU partition
# Note: This script runs GPU methods (Autoencoder, GAIN)
#       For CPU-only methods, use run_simulation_hpc_cpu.sh
#       GPU methods are slow, so we limit parallel runs to 4 to avoid GPU contention

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Load modules if needed (adjust for your HPC)
# module load python/3.9
# module load cuda/11.8
source ~/.bashrc
conda activate CSE595

cd /home/yhongda/include_y_mvi/include-outcome-in-mvi
# Set Python path (if needed)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Set number of processes based on SLURM allocation
NUM_PROCESSES=${SLURM_CPUS_PER_TASK:-8}
export NUM_PROCESSES

# Limit parallel runs to reduce GPU contention
export MAX_PARALLEL_RUNS=4

# Set method filter to GPU-only
export METHOD_FILTER=gpu

# Configuration file (change this to your desired config)
CONFIG_FILE="${1:-config_full_factorial.json}"

echo "Running GPU-only simulation with config: $CONFIG_FILE"
echo "Using $NUM_PROCESSES parallel processes (parallelizing across runs)"
echo "Method filter: GPU-only (Autoencoder, GAIN)"
echo "Max parallel runs: $MAX_PARALLEL_RUNS (to reduce GPU contention)"
echo ""
echo "Parallelization strategy:"
echo "  - Single parameter combination detected"
echo "  - Runs are parallelized across $MAX_PARALLEL_RUNS processes (limited for GPU)"
echo "  - GPU methods: Autoencoder and GAIN require GPU"
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
        
        # Get total allocated CPUs
        TOTAL_CORES=$SLURM_CPUS_PER_TASK
        
        # Get number of active Python processes
        ACTIVE_PROCS=$(ps -ef | grep python | grep -v grep | wc -l)
        
        # Estimate active and idle cores based on CPU usage
        if [ "$CPU_USAGE" != "N/A" ]; then
            ACTIVE_CORES=$(echo "$TOTAL_CORES $CPU_USAGE" | awk '{printf "%.1f", $1 * ($2/100)}')
            IDLE_CORES=$(echo "$TOTAL_CORES $CPU_USAGE" | awk '{printf "%.1f", $1 * (1 - $2/100)}')
        else
            IDLE_CORES="N/A"
            ACTIVE_CORES="N/A"
        fi
        
        echo "$TIMESTAMP,$CPU_USAGE,$IDLE_CORES,$ACTIVE_CORES,$TOTAL_CORES,$ACTIVE_PROCS"
        sleep 60  # Update every 60 seconds
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

# Get method filter from environment
method_filter = os.environ.get('METHOD_FILTER', 'gpu')
print(f'Running simulation with method_filter: {method_filter}')

# Run simulation
try:
    results_all, results_avg = run_simulation(config_file='$CONFIG_FILE', method_filter=method_filter)
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

