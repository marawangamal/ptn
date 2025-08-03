#!/usr/bin/env bash
# tmux_cockpit.sh
#
# Usage:
#   ./tmux_cockpit.sh python train.py --epochs 10
#
# Optional flags BEFORE the work command:
#   --monitor "<cmd>"  override monitor command (default: combined CPU/GPU monitor)
#   --session NAME     tmux session name (default: dev)

set -euo pipefail

# ---- defaults ----------------------------------------------------------------
# Clean monitoring command with peak tracking
MONITOR_CMD="bash -c 'max_cpu=0; max_ram=0; declare -a max_gpu; declare -a max_vram; declare -a max_temp; while true; do clear; echo \"📊 System Monitor - \$(date +\"%H:%M:%S\")\"; echo; cpu=\$(top -bn1 | grep \"Cpu(s)\" | awk \"{print \\\$2}\" | cut -d\"%\" -f1 | cut -d\".\" -f1); ram=\$(free | awk \"NR==2{printf \\\"%.0f\\\", \\\$3/\\\$2*100}\"); if (( cpu > max_cpu )); then max_cpu=\$cpu; fi; if (( ram > max_ram )); then max_ram=\$ram; fi; echo \"💻 CPU: \${cpu}% (peak: \${max_cpu}%)\"; echo \"🧠 RAM: \${ram}% (peak: \${max_ram}%)\"; echo; echo \"🎮 GPUs:\"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=\",\" read -r idx gpu_util mem_used mem_total temp; do vram_pct=\$(awk \"BEGIN{printf \\\"%.0f\\\", \$mem_used/\$mem_total*100}\"); if [[ -z \"\${max_gpu[\$idx]}\" ]] || (( gpu_util > \${max_gpu[\$idx]:-0} )); then max_gpu[\$idx]=\$gpu_util; fi; if [[ -z \"\${max_vram[\$idx]}\" ]] || (( vram_pct > \${max_vram[\$idx]:-0} )); then max_vram[\$idx]=\$vram_pct; fi; if [[ -z \"\${max_temp[\$idx]}\" ]] || (( temp > \${max_temp[\$idx]:-0} )); then max_temp[\$idx]=\$temp; fi; printf \"  GPU\$idx: %2s%% | VRAM: %2s%% | %s°C (peaks: %s%%/%s%%/%s°C)\\n\" \"\$gpu_util\" \"\$vram_pct\" \"\$temp\" \"\${max_gpu[\$idx]:-0}\" \"\${max_vram[\$idx]:-0}\" \"\${max_temp[\$idx]:-0}\"; done; sleep 2; done'"
SESSION="dev"

# ---- parse overrides ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --monitor)  MONITOR_CMD="$2"; shift 2 ;;
    --session)  SESSION="$2";     shift 2 ;;
    --)         shift; break ;;             # explicit end of flags
    *)          break ;;                    # first non-flag = start of work cmd
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Error: you must supply the command to run in the left pane."
  echo "Example: $0 python train.py --epochs 10"
  echo ""
  echo "Available monitor overrides:"
  echo "  --monitor 'htop'                    # CPU only"
  echo "  --monitor 'watch -n1 nvidia-smi'   # GPU only"
  echo "  --monitor 'nvtop'                  # GPU interactive"
  exit 1
fi
WORK_CMD="$*"

# ---- launch / attach ---------------------------------------------------------
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux-cockpit] Attaching to existing session '$SESSION'..."
  tmux attach -t "$SESSION"
  exit
fi

echo "[tmux-cockpit] Creating session '$SESSION'..."
tmux new-session  -d -s "$SESSION"              # pane 0 (left - work pane)
tmux send-keys    -t "$SESSION":0.0 "$WORK_CMD" C-m

tmux split-window -h -p 20                      # pane 1 (right - monitor pane)
tmux send-keys    -t "$SESSION":0.1 "$MONITOR_CMD" C-m

tmux select-pane  -t "$SESSION":0.0             # put focus back on work pane
tmux attach       -t "$SESSION"