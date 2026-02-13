#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${ROOT}/src/hybrid_nav_robot/experiments/results"
METRICS_SRC="${ROOT}/src/hybrid_nav_robot/experiments/latest_metrics.json"

mkdir -p "${OUT_DIR}"

goals=(
  "6.0 2.0"
  "8.0 0.0"
  "6.0 -2.0"
)
planner_modes=("rule" "rl")

run_seconds=40

for mode in "${planner_modes[@]}"; do
  for goal in "${goals[@]}"; do
    gx=$(echo "${goal}" | awk '{print $1}')
    gy=$(echo "${goal}" | awk '{print $2}')
    ts=$(date +%Y%m%d_%H%M%S)
    out_file="${OUT_DIR}/run_${ts}_mode_${mode}_goal_${gx}_${gy}.json"

    pkill -f "ros2 launch|gz sim|robot_state_publisher|parameter_bridge|planner_rule|spawner|controller_manager|hybrid_" || true
    sleep 2

    source /opt/ros/jazzy/setup.bash
    source "${ROOT}/install/setup.bash"

    timeout "${run_seconds}" \
      ros2 launch hybrid_nav_bringup hybrid_nav_demo.launch.py \
        planner_mode:="${mode}" goal_x:="${gx}" goal_y:="${gy}" enable_metrics:=true || true

    if [[ -f "${METRICS_SRC}" ]]; then
      cp "${METRICS_SRC}" "${out_file}"
      echo "Saved ${out_file}"
    else
      echo "No metrics found for goal (${gx}, ${gy})"
    fi
  done
done
