"""Ordinal 5-stage classification evaluation metrics (paper reproduction §12).

Labels are integers 1-5. Direct comparison with the paper centers on exact accuracy and
margin (within-1-stage); Macro F1 / QWK / Stage MAE are reported alongside as additional
ordinal metrics.
"""
from __future__ import annotations

import numpy as np

LABELS = [1, 2, 3, 4, 5]


def image_level_metrics(y_true, y_pred) -> dict:
    """Full metric bundle at image (or sample) level. y_true/y_pred are integer arrays 1-5."""
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    diff = np.abs(y_true - y_pred)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "exact_accuracy": float(accuracy_score(y_true, y_pred)),
        "within_1_stage_accuracy": float((diff <= 1).mean()),   # paper's 'margin 1 Stage'
        "within_half_stage_accuracy": float((diff <= 0.5).mean()),  # equals exact accuracy for integer predictions
        "stage_mae": float(diff.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=LABELS)),
        "per_class_precision": [float(x) for x in prec],
        "per_class_recall": [float(x) for x in rec],
        "per_class_f1": [float(x) for x in f1],
        "per_class_support": [int(x) for x in support],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def summarize(m: dict) -> str:
    """One-line summary (for logging)."""
    return (f"acc={m['exact_accuracy']:.3f} within1={m['within_1_stage_accuracy']:.3f} "
            f"MAE={m['stage_mae']:.3f} macroF1={m['macro_f1']:.3f} QWK={m['qwk']:.3f}")
