#!/usr/bin/env bash
set -euo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_DIR="${WS_ROOT}/src/hybrid_nav_robot"
RESULTS_DIR="${REPO_DIR}/experiments/results"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"

if [[ -n "${RESUME_RUN_DIR}" ]]; then
  RUN_DIR="${RESUME_RUN_DIR%/}"
  STAMP="$(basename "${RUN_DIR}" | sed 's/^run_//')"
else
  RUN_DIR="${RESULTS_DIR}/run_${STAMP}"
fi

RAW_DIR="${RUN_DIR}/raw"
LOG_DIR="${RUN_DIR}/logs"

EPISODES_PER_RUN="${EPISODES_PER_RUN:-20}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-420}"
EPISODE_MAX_DURATION_SEC="${EPISODE_MAX_DURATION_SEC:-60.0}"
EPISODE_PROGRESS_DISTANCE="${EPISODE_PROGRESS_DISTANCE:-0.35}"
EPISODE_GOAL_PROGRESS_DISTANCE="${EPISODE_GOAL_PROGRESS_DISTANCE:-0.15}"
EPISODE_STUCK_TIMEOUT_SEC="${EPISODE_STUCK_TIMEOUT_SEC:-5.0}"
EPISODE_STUCK_MIN_CMD_LINEAR="${EPISODE_STUCK_MIN_CMD_LINEAR:-0.0}"
EPISODE_POST_RESET_GOAL_REARM_DISTANCE="${EPISODE_POST_RESET_GOAL_REARM_DISTANCE:-1.0}"
EPISODE_REARM_TIMEOUT_SEC="${EPISODE_REARM_TIMEOUT_SEC:-2.5}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
STRICT_COMPLETION="${STRICT_COMPLETION:-true}"
RL_POLICY_FILE="${RL_POLICY_FILE:-${REPO_DIR}/experiments/rl_policy.json}"
BENCHMARK_PLANNERS="${BENCHMARK_PLANNERS:-rule,rl}"
BENCHMARK_SCENARIOS="${BENCHMARK_SCENARIOS:-easy,medium,cluttered}"

mkdir -p "${RAW_DIR}" "${LOG_DIR}"
export MPLCONFIGDIR="${RUN_DIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"
export ROS_HOME="${ROS_HOME:-${RUN_DIR}/ros_home}"
mkdir -p "${ROS_HOME}/log"

# ROS setup scripts are not nounset-safe in all environments.
set +u
source /opt/ros/jazzy/setup.bash
source "${WS_ROOT}/install/setup.bash"
set -u

planner_modes=("rule" "rl")
scenarios=(
  "easy|6.0|2.0|1.40|2.2|true|true|11"
  "medium|8.0|0.0|1.60|2.4|true|true|23"
  "cluttered|6.0|-2.0|1.80|2.6|true|true|31"
)

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

matches_filter() {
  local value="$1"
  local filter_csv="${2,,}"
  if [[ -z "${filter_csv}" || "${filter_csv}" == "all" ]]; then
    return 0
  fi
  [[ ",${filter_csv}," == *",${value},"* ]]
}

cleanup_processes() {
  # Avoid broad "hybrid_" match because it can kill this script path itself.
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

count_episodes() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "0"
    return
  fi
  python3 - "$file" <<'PY'
import json
import sys
path = sys.argv[1]
count = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("reason") in ("goal", "stuck", "timeout"):
            count += 1
print(count)
PY
}

trap cleanup_processes EXIT

echo "Benchmark run dir: ${RUN_DIR}"
if [[ -n "${RESUME_RUN_DIR}" ]]; then
  echo "Resume mode: enabled"
fi

declare -a incomplete_runs=()

for mode in "${planner_modes[@]}"; do
  if ! matches_filter "${mode}" "${BENCHMARK_PLANNERS}"; then
    continue
  fi
  for spec in "${scenarios[@]}"; do
    IFS='|' read -r scenario goal_x goal_y avoid_gain reverse_duration randomize_start randomize_obstacles seed <<<"${spec}"
    if ! matches_filter "${scenario}" "${BENCHMARK_SCENARIOS}"; then
      continue
    fi
    episode_file="${RAW_DIR}/episodes_${scenario}_${mode}.jsonl"
    latest_file="${RAW_DIR}/latest_metrics_${scenario}_${mode}.json"
    launch_log="${LOG_DIR}/launch_${scenario}_${mode}.log"

    existing_episodes="$(count_episodes "${episode_file}")"
    if is_true "${SKIP_COMPLETED}" && (( existing_episodes >= EPISODES_PER_RUN )); then
      echo "[SKIP] scenario=${scenario} planner=${mode} already complete (${existing_episodes}/${EPISODES_PER_RUN})"
      continue
    fi

    if (( existing_episodes == 0 )); then
      rm -f "${episode_file}" "${latest_file}" "${launch_log}"
    else
      echo "[RESUME] scenario=${scenario} planner=${mode} from ${existing_episodes}/${EPISODES_PER_RUN} episodes"
    fi
    echo "[RUN] scenario=${scenario} planner=${mode} episodes=${EPISODES_PER_RUN}"
    cleanup_processes
    sleep 2

    ros2 launch hybrid_nav_bringup hybrid_nav_demo.launch.py \
      planner_mode:="${mode}" \
      rl_policy_file:="${RL_POLICY_FILE}" \
      benchmark_scenario_label:="${scenario}" \
      goal_x:="${goal_x}" goal_y:="${goal_y}" reach_tolerance:=0.15 \
      stop_distance:=0.50 hard_stop_distance:=0.20 \
      avoid_distance:=1.20 avoid_gain:="${avoid_gain}" \
      reverse_enabled:=true reverse_trigger_distance:=0.32 \
      reverse_speed:=0.40 reverse_turn_speed:=0.70 reverse_duration_sec:="${reverse_duration}" \
      escape_forward_duration_sec:=1.8 escape_forward_speed:=0.40 \
      auto_reset_enabled:=true reset_on_goal:=true reset_on_stuck:=true \
      reset_mode:=model_only reset_pause_sec:=0.6 \
      episode_progress_distance:="${EPISODE_PROGRESS_DISTANCE}" \
      episode_goal_progress_distance:="${EPISODE_GOAL_PROGRESS_DISTANCE}" \
      episode_stuck_timeout_sec:="${EPISODE_STUCK_TIMEOUT_SEC}" \
      episode_stuck_min_cmd_linear:="${EPISODE_STUCK_MIN_CMD_LINEAR}" \
      episode_max_duration_sec:="${EPISODE_MAX_DURATION_SEC}" \
      episode_post_reset_goal_rearm_distance:="${EPISODE_POST_RESET_GOAL_REARM_DISTANCE}" \
      episode_rearm_timeout_sec:="${EPISODE_REARM_TIMEOUT_SEC}" \
      episode_fallback_all_reset_on_rearm_timeout:=true \
      episode_wait_for_controllers_active:=true \
      episode_randomization_enabled:=true \
      episode_random_seed:="${seed}" \
      episode_randomize_start_pose:="${randomize_start}" \
      episode_randomize_goal:=false \
      episode_randomize_obstacles:="${randomize_obstacles}" \
      enable_dataset_collection:=false \
      enable_metrics:=true \
      metrics_output_file:="${latest_file}" \
      metrics_append_episode_summary:=true \
      metrics_episode_summary_file:="${episode_file}" \
      > "${launch_log}" 2>&1 &
    launch_pid=$!

    started_at="$(date +%s)"
    combo_status="ok"
    while true; do
      sleep 2
      episodes_done="$(count_episodes "${episode_file}")"
      now_ts="$(date +%s)"
      elapsed="$((now_ts - started_at))"
      echo "  progress: ${episodes_done}/${EPISODES_PER_RUN} episodes, elapsed=${elapsed}s"
      if (( episodes_done >= EPISODES_PER_RUN )); then
        echo "  completed ${scenario}/${mode}"
        break
      fi
      if (( elapsed >= MAX_WAIT_SEC )); then
        echo "  timeout for ${scenario}/${mode}, partial episodes=${episodes_done}" >&2
        combo_status="timeout"
        break
      fi
      if ! kill -0 "${launch_pid}" 2>/dev/null; then
        echo "  launch exited early for ${scenario}/${mode}" >&2
        combo_status="launch_exited"
        break
      fi
    done

    kill "${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
    cleanup_processes
    sleep 2

    episodes_done="$(count_episodes "${episode_file}")"
    if (( episodes_done < EPISODES_PER_RUN )); then
      incomplete_runs+=("${scenario}/${mode}:${episodes_done}/${EPISODES_PER_RUN}:${combo_status}")
    fi
  done
done

if ls "${RAW_DIR}"/episodes_*.jsonl >/dev/null 2>&1; then
  python3 "${REPO_DIR}/experiments/summarize_benchmark.py" \
    --input-dir "${RAW_DIR}" \
    --output-dir "${RESULTS_DIR}" \
    --run-dir "${RUN_DIR}" \
    --run-id "${STAMP}"
else
  echo "[ERROR] no episode summary files found in ${RAW_DIR}" >&2
  exit 2
fi

if ((${#incomplete_runs[@]} > 0)); then
  echo
  echo "[WARN] Incomplete scenario/planner runs:"
  for item in "${incomplete_runs[@]}"; do
    echo "  - ${item}"
  done
  if is_true "${STRICT_COMPLETION}"; then
    echo "[ERROR] STRICT_COMPLETION=true, returning non-zero status." >&2
    exit 3
  fi
fi

echo
echo "Benchmark finished."
echo "Run directory: ${RUN_DIR}"
echo "Summary CSV : ${RESULTS_DIR}/benchmark_summary.csv"
echo "Summary JSON: ${RESULTS_DIR}/benchmark_summary.json"
echo "Plots       : ${RESULTS_DIR}/success_rate.png, ${RESULTS_DIR}/time_to_goal.png, ${RESULTS_DIR}/safety_hits.png"
echo "Launch logs : ${LOG_DIR}"
echo
echo "To inspect logs:"
for mode in "${planner_modes[@]}"; do
  for spec in "${scenarios[@]}"; do
    IFS='|' read -r scenario _ <<<"${spec}"
    echo "  ${LOG_DIR}/launch_${scenario}_${mode}.log"
  done
done
