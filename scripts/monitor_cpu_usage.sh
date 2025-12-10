#!/bin/bash
# Monitor CPU usage for a running SLURM job
# Usage: ./scripts/monitor_cpu_usage.sh <job_id> [interval_seconds]

JOB_ID=$1
INTERVAL=${2:-5}  # Default 5 seconds

if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 <job_id> [interval_seconds]"
    echo "Example: $0 12345678 10"
    exit 1
fi

# Get the node where the job is running
NODE=$(squeue -j $JOB_ID -h -o "%N" | head -1)

if [ -z "$NODE" ]; then
    echo "Job $JOB_ID not found or not running"
    exit 1
fi

echo "Monitoring CPU usage for job $JOB_ID on node $NODE"
echo "Press Ctrl+C to stop"
echo "=========================================="
printf "%-19s %8s %12s %12s %12s\n" "Timestamp" "CPU%" "Idle Cores" "Active Cores" "Total Cores"
echo "=========================================="

while true; do
    # Get CPU usage from the node
    if [ "$NODE" != "N/A" ] && [ ! -z "$NODE" ]; then
        # SSH to node and get CPU stats (if you have SSH access)
        # Otherwise, use sstat if available
        if command -v sstat &> /dev/null; then
            CPU_STATS=$(sstat -j $JOB_ID --format=CPU -n -P 2>/dev/null | tail -1)
            if [ ! -z "$CPU_STATS" ]; then
                CPU_USAGE=$(echo $CPU_STATS | awk '{print $1}')
            else
                CPU_USAGE="N/A"
            fi
        else
            # Fallback: estimate from process count
            CPU_USAGE="N/A"
        fi
    fi
    
    # Alternative: Check from the job's output/logs
    # Count active Python processes
    ACTIVE_PROCS=$(ps aux | grep -E "python.*run_simulation|python -c.*run_simulation" | grep -v grep | wc -l)
    
    # Get allocated CPUs
    ALLOCATED_CPUS=$(squeue -j $JOB_ID -h -o "%C" 2>/dev/null)
    if [ -z "$ALLOCATED_CPUS" ]; then
        ALLOCATED_CPUS="N/A"
    fi
    
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Try to get actual CPU usage using top (if we can access the node)
    # For now, show process count as proxy
    printf "%-19s %8s %12s %12s %12s\n" "$TIMESTAMP" "${CPU_USAGE:-N/A}" "${IDLE_CORES:-N/A}" "$ACTIVE_PROCS" "$ALLOCATED_CPUS"
    
    sleep $INTERVAL
done

