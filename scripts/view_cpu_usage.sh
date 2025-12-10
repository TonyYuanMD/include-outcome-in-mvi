#!/bin/bash
# View CPU usage summary from monitoring log
# Usage: ./scripts/view_cpu_usage.sh [log_file]

LOG_FILE=${1:-"log/cpu_monitor_*.log"}

# Find the most recent log file
LATEST_LOG=$(ls -t $LOG_FILE 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ] || [ ! -f "$LATEST_LOG" ]; then
    echo "No CPU monitoring log found. Looking for: $LOG_FILE"
    echo "Make sure the job has run with CPU monitoring enabled."
    exit 1
fi

echo "CPU Usage Summary from: $LATEST_LOG"
echo "=========================================="
echo ""

# Skip header and calculate statistics
if [ $(wc -l < "$LATEST_LOG") -gt 1 ]; then
    echo "Statistics (excluding header):"
    tail -n +2 "$LATEST_LOG" | awk -F',' '
    {
        if ($2 != "N/A" && $2 != "") {
            cpu_sum += $2
            cpu_count++
            if (cpu_count == 1 || $2 < cpu_min) cpu_min = $2
            if ($2 > cpu_max) cpu_max = $2
        }
        if ($3 != "N/A" && $3 != "") {
            idle_sum += $3
            idle_count++
        }
        if ($4 != "N/A" && $4 != "") {
            active_sum += $4
            active_count++
        }
    }
    END {
        if (cpu_count > 0) {
            printf "  Average CPU Usage: %.1f%%\n", cpu_sum/cpu_count
            printf "  Min CPU Usage:    %.1f%%\n", cpu_min
            printf "  Max CPU Usage:    %.1f%%\n", cpu_max
        } else {
            print "  CPU Usage: N/A (no valid data)"
        }
        if (idle_count > 0) {
            printf "  Average Idle Cores: %.1f\n", idle_sum/idle_count
        }
        if (active_count > 0) {
            printf "  Average Active Cores: %.1f\n", active_sum/active_count
        }
    }'
    echo ""
    echo "Recent activity (last 5 entries):"
    echo "----------------------------------------"
    tail -5 "$LATEST_LOG" | column -t -s','
else
    echo "Log file is empty or only contains header."
fi

