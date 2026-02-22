from __future__ import annotations

import json
import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _default_dataset_path() -> str:
    env_root = os.environ.get("HYBRID_NAV_ROBOT_ROOT", "").strip()
    if env_root:
        path = Path(env_root).expanduser().resolve()
        if path.exists():
            return str(path / "experiments" / "semantic_rl_dataset.jsonl")

    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "experiments").exists():
            return str(parent / "experiments" / "semantic_rl_dataset.jsonl")

    for parent in here.parents:
        if parent.name == "install":
            candidate = parent.parent / "src" / "hybrid_nav_robot" / "experiments" / "semantic_rl_dataset.jsonl"
            if candidate.parent.exists():
                return str(candidate)

    return str(Path.cwd() / "experiments" / "semantic_rl_dataset.jsonl")


class SemanticDatasetCollectorNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_dataset_collector")

        self.declare_parameter("scores_topic", "/semantic_rl/candidate_scores_json")
        self.declare_parameter("status_topic", "/exploration/status")
        self.declare_parameter("output_file", os.environ.get("SEMANTIC_RL_DATASET", _default_dataset_path()))
        self.declare_parameter("scenario_label", "house_sim")
        self.declare_parameter("flush_every", 50)

        scores_topic = str(self.get_parameter("scores_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        output_file = str(self.get_parameter("output_file").value)
        self.scenario_label = str(self.get_parameter("scenario_label").value)
        self.flush_every = int(self.get_parameter("flush_every").value)

        self.out_path = Path(output_file)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.out_path.open("a", encoding="utf-8")

        self.write_count = 0
        self.last_status_event = ""

        self.create_subscription(String, scores_topic, self.on_scores, 10)
        self.create_subscription(String, status_topic, self.on_status, 10)

        self.get_logger().info(f"semantic_dataset_collector writing to {self.out_path}")

    def on_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.last_status_event = str(payload.get("event", ""))

    def on_scores(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        candidates = payload.get("candidates", [])
        stamp_sec = float(payload.get("stamp_sec", self.get_clock().now().nanoseconds / 1e9))
        if not isinstance(candidates, list):
            return

        for idx, cand in enumerate(candidates):
            features = cand.get("features", {})
            record = {
                "stamp_sec": stamp_sec,
                "scenario": self.scenario_label,
                "candidate_rank": idx,
                "selected": idx == 0,
                "x": float(cand.get("x", 0.0)),
                "y": float(cand.get("y", 0.0)),
                "score": float(cand.get("score", 0.0)),
                "pseudo_expert_score": float(cand.get("pseudo_expert_score", 0.0)),
                "distance_to_frontier": float(features.get("distance_to_frontier", 0.0)),
                "estimated_free_gain": float(features.get("estimated_free_gain", 0.0)),
                "heading_change": float(features.get("heading_change", 0.0)),
                "local_obstacle_risk": float(features.get("local_obstacle_risk", 0.0)),
                "semantic_novelty_score": float(features.get("semantic_novelty_score", 0.0)),
                "semantic_priority_score": float(features.get("semantic_priority_score", 0.0)),
                "latest_status_event": self.last_status_event,
            }
            self.stream.write(json.dumps(record) + "\n")
            self.write_count += 1

        if self.write_count % max(1, self.flush_every) == 0:
            self.stream.flush()

    def destroy_node(self) -> bool:
        try:
            self.stream.flush()
            self.stream.close()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = SemanticDatasetCollectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
