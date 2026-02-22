from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .scoring_utils import FEATURE_ORDER


def _detect_repo_root() -> Path:
    env_root = os.environ.get("HYBRID_NAV_ROBOT_ROOT", "").strip()
    if env_root:
        path = Path(env_root).expanduser().resolve()
        if path.exists():
            return path

    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "experiments").exists():
            return parent

    for parent in here.parents:
        if parent.name == "install":
            candidate = parent.parent / "src" / "hybrid_nav_robot"
            if candidate.exists():
                return candidate

    return Path.cwd()


def _default_dataset() -> Path:
    return _detect_repo_root() / "experiments" / "semantic_rl_dataset.jsonl"


def _default_output() -> Path:
    return _detect_repo_root() / "experiments" / "semantic_rl_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train offline semantic RL frontier ranking policy")
    parser.add_argument("--dataset", default=str(_default_dataset()))
    parser.add_argument("--output", default=str(_default_output()))
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--chosen-weight", type=float, default=1.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(raw)

    if len(rows) < args.min_samples:
        raise RuntimeError(f"Not enough samples ({len(rows)}). Need at least {args.min_samples}.")

    x_list = []
    y_list = []
    w_list = []
    for row in rows:
        vec = [float(row.get(k, 0.0)) for k in FEATURE_ORDER]
        target = float(row.get("pseudo_expert_score", 0.0))
        selected = bool(row.get("selected", False))
        status = str(row.get("latest_status_event", ""))

        weight = args.chosen_weight if selected else 1.0
        if status == "goal_succeeded":
            weight *= 1.25
        elif status in {"goal_aborted", "timeout"}:
            weight *= 0.9

        x_list.append(vec)
        y_list.append(target)
        w_list.append(weight)

    x = np.asarray(x_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64)
    w = np.asarray(w_list, dtype=np.float64)

    # Weighted ridge regression with explicit bias term.
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    xb = np.hstack([ones, x])
    w_diag = np.diag(w)
    reg = np.eye(xb.shape[1], dtype=np.float64) * float(args.l2)
    reg[0, 0] = 0.0

    lhs = xb.T @ w_diag @ xb + reg
    rhs = xb.T @ w_diag @ y
    coeff = np.linalg.solve(lhs, rhs)

    bias = float(coeff[0])
    weights = {name: float(coeff[i + 1]) for i, name in enumerate(FEATURE_ORDER)}

    pred = xb @ coeff
    mse = float(np.mean((pred - y) ** 2))

    policy = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_method": "offline_weighted_ridge",
        "dataset_path": str(dataset_path),
        "sample_count": int(x.shape[0]),
        "feature_order": FEATURE_ORDER,
        "bias": bias,
        "weights": weights,
        "cooldown_sec": 2.0,
        "stats": {
            "target_mean": float(np.mean(y)),
            "target_std": float(np.std(y)),
            "prediction_mse": mse,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"Saved semantic RL policy to {out}")
    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    main()
