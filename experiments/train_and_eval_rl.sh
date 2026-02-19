#!/usr/bin/env bash
set -euo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_DIR="${WS_ROOT}/src/hybrid_nav_robot"
RESULTS_DIR="${REPO_DIR}/experiments/results"
DATASET_FILE="${DATASET_FILE:-${REPO_DIR}/experiments/rl_dataset.jsonl}"
POLICY_FILE="${POLICY_FILE:-${REPO_DIR}/experiments/rl_policy.json}"
PACKAGE_POLICY_FILE="${PACKAGE_POLICY_FILE:-${REPO_DIR}/hybrid_nav_rl_planner/config/rl_policy.json}"

COLLECT_DURATION_SEC="${COLLECT_DURATION_SEC:-0}"
RUN_BENCHMARK="${RUN_BENCHMARK:-true}"
EPISODES_PER_RUN="${EPISODES_PER_RUN:-20}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-420}"
ANALYZE_BASELINE="${ANALYZE_BASELINE:-true}"
SYNC_POLICY_CONFIG="${SYNC_POLICY_CONFIG:-true}"
DO_BUILD="${DO_BUILD:-true}"
RL_REWARD_PRESET="${RL_REWARD_PRESET:-uplift_v1}"
RESET_DATASET_ON_COLLECT="${RESET_DATASET_ON_COLLECT:-true}"

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

has_episode_files() {
  local run_dir="$1"
  [[ -d "${run_dir}/raw" ]] && ls "${run_dir}/raw"/episodes_*.jsonl >/dev/null 2>&1
}

latest_run_with_episodes() {
  local d
  for d in $(ls -dt "${RESULTS_DIR}"/run_* 2>/dev/null); do
    if has_episode_files "${d}"; then
      echo "${d}"
      return 0
    fi
  done
  return 1
}

print_dataset_scenario_mix() {
  local dataset="$1"
  python3 - "$dataset" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
counter = Counter()
if not path.exists():
    print("[WARN] dataset file not found for scenario-mix check:", path)
    sys.exit(0)

with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        scenario = str(row.get("scenario", "default")).strip() or "default"
        counter[scenario] += 1

total = sum(counter.values())
if total == 0:
    print("[WARN] dataset is empty:", path)
    sys.exit(0)

print("[INFO] Dataset scenario mix:")
for name, count in sorted(counter.items()):
    ratio = count / total
    print(f"  - {name}: {count} samples ({ratio:.1%})")

if len(counter) == 1 and "default" in counter:
    print("[WARN] Dataset only contains 'default' scenario label. RL balancing across scenarios is disabled.")
PY
}

cleanup() {
  pkill -f "ros2 launch hybrid_nav_bringup hybrid_nav_demo.launch.py" || true
  pkill -f "gz sim" || true
  pkill -f "parameter_bridge" || true
  pkill -f "controller_manager" || true
  pkill -f "spawner" || true
  pkill -f "ackermann_control_node" || true
  pkill -f "robot_state_publisher" || true
  pkill -f "hybrid_rule_planner" || true
  pkill -f "hybrid_rl_planner" || true
  pkill -f "hybrid_safety_layer" || true
  pkill -f "hybrid_metrics_logger" || true
  pkill -f "hybrid_episode_manager" || true
  pkill -f "hybrid_rl_dataset_collector" || true
}

trap cleanup EXIT

mkdir -p "${RESULTS_DIR}"
export ROS_HOME="${ROS_HOME:-${RESULTS_DIR}/ros_home}"
mkdir -p "${ROS_HOME}/log"

set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "${WS_ROOT}/install/setup.bash" ]]; then
  source "${WS_ROOT}/install/setup.bash"
fi
set -u

if is_true "${DO_BUILD}"; then
  echo "[STEP] Build updated packages"
  colcon build --base-paths "${WS_ROOT}/src" --packages-select \
    hybrid_nav_bringup \
    hybrid_nav_metrics \
    hybrid_nav_rl_planner \
    hybrid_nav_rule_planner \
    hybrid_nav_safety_layer
  set +u
  source "${WS_ROOT}/install/setup.bash"
  set -u
fi

echo "[INFO] WS_ROOT=${WS_ROOT}"
echo "[INFO] DATASET_FILE=${DATASET_FILE}"
echo "[INFO] POLICY_FILE=${POLICY_FILE}"
echo "[INFO] RL_REWARD_PRESET=${RL_REWARD_PRESET}"

baseline_run="$(latest_run_with_episodes || true)"

if is_true "${ANALYZE_BASELINE}" && [[ -n "${baseline_run}" ]]; then
  if has_episode_files "${baseline_run}"; then
    echo "[STEP] Analyze baseline RL failures from: ${baseline_run}"
    python3 "${REPO_DIR}/experiments/analyze_rl_failures.py" \
      --input "${baseline_run}" \
      --planner-mode rl \
      --output "${baseline_run}/rl_failure_analysis_baseline.json" || true
  else
    echo "[WARN] Baseline run has no episode files, skipping baseline analysis: ${baseline_run}"
  fi
fi

if (( COLLECT_DURATION_SEC > 0 )); then
  echo "[STEP] Collect multi-scenario dataset with rule planner for ${COLLECT_DURATION_SEC}s"
  if is_true "${RESET_DATASET_ON_COLLECT}" && [[ -f "${DATASET_FILE}" ]]; then
    backup="${DATASET_FILE}.bak_$(date +%Y%m%d_%H%M%S)"
    mv "${DATASET_FILE}" "${backup}"
    echo "[INFO] Existing dataset moved to ${backup}"
  fi
  scenario_specs=(
    "easy|6.0|2.0|1.40|2.2|true|true|11"
    "medium|8.0|0.0|1.60|2.4|true|true|23"
    "cluttered|6.0|-2.0|1.80|2.6|true|true|31"
  )
  n="${#scenario_specs[@]}"
  base_duration="$((COLLECT_DURATION_SEC / n))"
  remainder="$((COLLECT_DURATION_SEC % n))"
  for idx in "${!scenario_specs[@]}"; do
    IFS='|' read -r scenario goal_x goal_y avoid_gain reverse_duration randomize_start randomize_obstacles seed <<<"${scenario_specs[$idx]}"
    duration="${base_duration}"
    if (( idx < remainder )); then
      duration="$((duration + 1))"
    fi
    if (( duration <= 0 )); then
      continue
    fi
    echo "[STEP] Collect ${scenario} for ${duration}s"
    cleanup
    sleep 2
    ros2 launch hybrid_nav_bringup hybrid_nav_demo.launch.py \
      planner_mode:=rule \
      benchmark_scenario_label:="${scenario}" \
      goal_x:="${goal_x}" goal_y:="${goal_y}" reach_tolerance:=0.15 \
      stop_distance:=0.50 hard_stop_distance:=0.20 \
      avoid_distance:=1.20 avoid_gain:="${avoid_gain}" \
      reverse_enabled:=true reverse_trigger_distance:=0.32 \
      reverse_speed:=0.40 reverse_turn_speed:=0.70 reverse_duration_sec:="${reverse_duration}" \
      escape_forward_duration_sec:=1.8 escape_forward_speed:=0.40 \
      auto_reset_enabled:=true reset_on_goal:=true reset_on_stuck:=true \
      reset_mode:=model_only reset_pause_sec:=0.6 \
      episode_post_reset_goal_rearm_distance:=1.0 \
      episode_rearm_timeout_sec:=2.5 \
      episode_fallback_all_reset_on_rearm_timeout:=true \
      episode_wait_for_controllers_active:=true \
      episode_randomization_enabled:=true \
      episode_random_seed:="${seed}" \
      episode_randomize_start_pose:="${randomize_start}" \
      episode_randomize_goal:=false \
      episode_randomize_obstacles:="${randomize_obstacles}" \
      enable_dataset_collection:=true \
      dataset_output_file:="${DATASET_FILE}" \
      dataset_cmd_topic:=/cmd_vel \
      enable_metrics:=false \
      > "${RESULTS_DIR}/collect_rl_data_${scenario}.log" 2>&1 &
    launch_pid=$!
    sleep "${duration}"
    kill "${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
    cleanup
    sleep 2
  done
fi

print_dataset_scenario_mix "${DATASET_FILE}"

echo "[STEP] Train reward-weighted RL policy"
case "${RL_REWARD_PRESET}" in
  uplift_v1)
    reward_progress_scale="10.5"
    reward_goal_bonus="9.5"
    penalty_collision="1.4"
    penalty_stuck_step="0.10"
    penalty_stuck_terminal="2.6"
    penalty_angular_oscillation="0.04"
    success_episode_weight="2.2"
    k_linear_min="0.40"
    k_heading_min="0.75"
    k_avoid_min="0.20"
    k_front_brake_max="0.60"
    ;;
  *)
    echo "[ERROR] Unknown RL_REWARD_PRESET=${RL_REWARD_PRESET}" >&2
    exit 2
    ;;
esac

python3 "${REPO_DIR}/hybrid_nav_rl_planner/hybrid_nav_rl_planner/train_stub.py" \
  --dataset "${DATASET_FILE}" \
  --output "${POLICY_FILE}" \
  --preset-label "${RL_REWARD_PRESET}" \
  --reward-progress-scale "${reward_progress_scale}" \
  --reward-goal-bonus "${reward_goal_bonus}" \
  --penalty-collision "${penalty_collision}" \
  --penalty-stuck-step "${penalty_stuck_step}" \
  --penalty-stuck-terminal "${penalty_stuck_terminal}" \
  --penalty-angular-oscillation "${penalty_angular_oscillation}" \
  --success-episode-weight "${success_episode_weight}" \
  --k-linear-min "${k_linear_min}" \
  --k-heading-min "${k_heading_min}" \
  --k-avoid-min "${k_avoid_min}" \
  --k-front-brake-max "${k_front_brake_max}" \
  --balance-by-scenario true

if is_true "${SYNC_POLICY_CONFIG}"; then
  cp "${POLICY_FILE}" "${PACKAGE_POLICY_FILE}"
  echo "[INFO] Synced policy to ${PACKAGE_POLICY_FILE}"
fi

if is_true "${RUN_BENCHMARK}"; then
  echo "[STEP] Run benchmark matrix"
  EPISODES_PER_RUN="${EPISODES_PER_RUN}" MAX_WAIT_SEC="${MAX_WAIT_SEC}" \
    "${REPO_DIR}/experiments/run_benchmark.sh"
fi

latest_run="$(latest_run_with_episodes || true)"

if [[ -n "${latest_run}" ]]; then
  if has_episode_files "${latest_run}"; then
    echo "[STEP] Analyze RL failures from latest run: ${latest_run}"
    python3 "${REPO_DIR}/experiments/analyze_rl_failures.py" \
      --input "${latest_run}" \
      --planner-mode rl \
      --output "${latest_run}/rl_failure_analysis_post_uplift.json" || true
  else
    echo "[WARN] Latest run has no episode files, skipping post-uplift analysis: ${latest_run}"
  fi
fi

echo
echo "Done."
echo "Policy      : ${POLICY_FILE}"
echo "Latest run  : ${latest_run}"
echo "Summary JSON: ${RESULTS_DIR}/benchmark_summary.json"
echo "Summary CSV : ${RESULTS_DIR}/benchmark_summary.csv"
