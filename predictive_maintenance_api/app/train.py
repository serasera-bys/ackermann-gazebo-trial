from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .features import FEATURE_NAMES
from .model import write_feature_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train predictive maintenance risk model")
    parser.add_argument("--data", default="data/cmapss_like.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        df = generate_synthetic_dataset(n_rows=2400, seed=args.seed)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    required = set(FEATURE_NAMES + ["failure_risk_label"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    x = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["failure_risk_label"].to_numpy(dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=args.seed,
        stratify=y,
    )

    model, algo = train_model(x_train, y_train, seed=args.seed)

    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    report = {
        "algorithm": algo,
        "rows": int(df.shape[0]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

    model_path = out_dir / "model.joblib"
    schema_path = out_dir / "feature_schema.json"
    eval_path = out_dir / "eval_report.json"
    meta_path = out_dir / "train_metadata.json"

    joblib.dump(model, model_path)
    write_feature_schema(str(schema_path))
    eval_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "algorithm": algo,
                "trained_rows": int(df.shape[0]),
                "feature_count": len(FEATURE_NAMES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved model: {model_path}")
    print(f"Saved schema: {schema_path}")
    print(f"Saved eval: {eval_path}")


def train_model(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[object, str]:
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model, "xgboost"
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(random_state=seed)
        model.fit(x_train, y_train)
        return model, "gradient_boosting"


def generate_synthetic_dataset(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        base = rng.normal(0.0, 1.0, size=5)
        drift = rng.normal(0.0, 0.3, size=5)
        noise = rng.normal(0.0, 0.2, size=5)

        means = base + 0.8 * drift
        stds = np.abs(rng.normal(0.3, 0.15, size=5))
        slopes = drift + rng.normal(0.0, 0.08, size=5)
        deltas = drift * rng.uniform(2.0, 4.0) + noise

        risk_score = (
            0.35 * np.mean(np.abs(deltas))
            + 0.25 * np.mean(stds)
            + 0.20 * np.mean(np.maximum(slopes, 0.0))
            + 0.20 * np.mean(np.maximum(means - 0.8, 0.0))
        )
        label = int(risk_score > 0.65)

        row = {
            "s1_mean": means[0],
            "s2_mean": means[1],
            "s3_mean": means[2],
            "s4_mean": means[3],
            "s5_mean": means[4],
            "s1_std": stds[0],
            "s2_std": stds[1],
            "s3_std": stds[2],
            "s4_std": stds[3],
            "s5_std": stds[4],
            "s1_slope": slopes[0],
            "s2_slope": slopes[1],
            "s3_slope": slopes[2],
            "s4_slope": slopes[3],
            "s5_slope": slopes[4],
            "delta_s1": deltas[0],
            "delta_s2": deltas[1],
            "delta_s3": deltas[2],
            "delta_s4": deltas[3],
            "delta_s5": deltas[4],
            "failure_risk_label": label,
        }
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
