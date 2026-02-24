import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from config import EvalConfig, RESULTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None,
) -> Dict[str, float]:
    if labels is None:
        labels = ["yes", "no", "maybe"]

    valid_mask = [p in labels for p in y_pred]
    n_valid = sum(valid_mask)
    n_total = len(y_pred)

    if n_valid == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0, "parse_rate": 0.0, "n_valid": 0, "n_total": n_total}

    yt = [y_true[i] for i in range(n_total) if valid_mask[i]]
    yp = [y_pred[i] for i in range(n_total) if valid_mask[i]]

    return {
        "accuracy": accuracy_score(yt, yp),
        "f1_macro": f1_score(yt, yp, labels=labels, average="macro", zero_division=0),
        "parse_rate": n_valid / n_total,
        "n_valid": n_valid,
        "n_total": n_total,
    }


def bootstrap_evaluate(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None,
    n_bootstrap: int = 1000,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    size = sample_size or n

    acc_scores = []
    f1_scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=size, replace=True)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        m = compute_metrics(yt, yp, labels)
        acc_scores.append(m["accuracy"])
        f1_scores.append(m["f1_macro"])

    def _summarize(scores):
        arr = np.array(scores)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    return {"accuracy": _summarize(acc_scores), "f1_macro": _summarize(f1_scores)}


def aggregate_random_trials(trial_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for key in trial_metrics[0]:
        vals = [m[key] for m in trial_metrics]
        metrics[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return metrics


def save_results(results: Dict, experiment_name: str, output_dir: Optional[Path] = None) -> Path:
    out_dir = output_dir or RESULTS_DIR
    out_path = out_dir / f"{experiment_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")
    return out_path


def print_comparison_table(results: Dict[str, Dict], labels: List[str] = None):
    print("\n" + "=" * 72)
    print(f"{'Method':<20} {'Accuracy':>10} {'F1 (macro)':>12} {'Parse %':>10}")
    print("-" * 72)

    for method, metrics in results.items():
        if isinstance(metrics.get("accuracy"), dict):
            acc = metrics["accuracy"]["mean"]
            f1 = metrics["f1_macro"]["mean"]
            acc_ci = f"±{metrics['accuracy']['std']:.3f}"
            f1_ci = f"±{metrics['f1_macro']['std']:.3f}"
            print(f"{method:<20} {acc:>7.4f}{acc_ci:>6} {f1:>8.4f}{f1_ci:>6}")
        else:
            acc = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_macro", 0)
            pr = metrics.get("parse_rate", 0)
            print(f"{method:<20} {acc:>10.4f} {f1:>12.4f} {pr:>10.1%}")

    print("=" * 72)