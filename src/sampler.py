"""Class imbalance handling -- paper-style random oversampling (§7).

Paper: "samples from underrepresented classifications were randomly duplicated to ensure
an equitable distribution ... random oversampling", applied only to the train set
(val/test keep their original distribution).

Implemented via actual index duplication (no image files are copied; dataset indices are
duplicated instead). Each class is randomly duplicated up to the count of the largest
class in train. Seed fixed.
"""
from __future__ import annotations

import numpy as np


def paper_oversample_indices(labels, seed: int = 42):
    """Return a paper-style randomly oversampled index list.

    labels: array of labels (1-5) for train samples. The returned indices are positional
    indices into labels (= train_df). Each class is randomly duplicated up to the max
    class count (the shortfall is filled by sampling with replacement).
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    target = counts.max()

    idx_all = np.arange(len(labels))
    out = []
    for c in classes:
        cls_idx = idx_all[labels == c]
        need = target - len(cls_idx)
        extra = rng.choice(cls_idx, size=need, replace=True) if need > 0 else np.array([], int)
        out.append(np.concatenate([cls_idx, extra]))
    result = np.concatenate(out)
    rng.shuffle(result)
    return result.tolist()


def effective_class_counts(labels, oversampled_index) -> dict:
    """Effective per-class sample counts after oversampling (for logging)."""
    labels = np.asarray(labels)
    sampled = labels[np.asarray(oversampled_index)]
    classes, counts = np.unique(sampled, return_counts=True)
    return {int(c): int(n) for c, n in zip(classes, counts)}
