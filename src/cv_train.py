"""K-fold cross-validation for the classifier (group-aware) — a more robust estimate than the
single fixed split.

Folds are built on the (Storage Group, Sample) key, NEVER image-level (CLAUDE.md §2.1), so the
same avocado never lands in both train and val within a fold. Trains the config's model once per
fold, records the best-val metric bundle per fold, and reports mean ± std across folds. Uses the
same curated data (metadata_clean.csv / manifest) and metrics (metrics.image_level_metrics) as the
single-split runs, so the numbers are comparable.

CV runs over ALL samples of the chosen dataset (the fixed train/val/test Split column is ignored;
k-fold is the evaluation here). A held-out final test would be a separate concern.

Run:
  python -m src.cv_train --config configs/paper/general_resnet18.yaml --folds 5
  python -m src.cv_train --config ... --folds 5 --epochs 20 --val-freq 50 --smoke-test
"""
from __future__ import annotations

import argparse

import numpy as np

try:
    from .dataset import DATASET_GROUPS, load_metadata_and_splits
    from .train import train
    from .utils import OUTPUT_DIR, load_config, save_json, set_seed
except ImportError:
    from dataset import DATASET_GROUPS, load_metadata_and_splits
    from train import train
    from utils import OUTPUT_DIR, load_config, save_json, set_seed

AGG_KEYS = ["exact_accuracy", "within_1_stage_accuracy", "stage_mae", "qwk", "macro_f1"]


def fold_val_keys(keys, k: int, seed: int) -> list[set]:
    """Partition unique group keys into k disjoint validation sets (seeded shuffle)."""
    if k < 2:
        raise ValueError("folds must be >= 2")
    arr = np.asarray(list(keys), dtype=object)
    perm = np.random.default_rng(seed).permutation(len(arr))
    return [set(arr[idx]) for idx in np.array_split(perm, k)]


def group_kfold(df, k: int, seed: int):
    """Yield (train_df, val_df) per fold, split on the (Storage Group, Sample) key (§2.1)."""
    key = df["Storage Group"].astype(str) + "_" + df["Sample"].astype(str)
    for val_keys in fold_val_keys(key.drop_duplicates().tolist(), k, seed):
        is_val = key.isin(val_keys)
        assert set(key[is_val]) & set(key[~is_val]) == set(), "group leakage in fold! (§2.1)"
        yield df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Group-aware k-fold CV for the classifier")
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs (per fold)")
    ap.add_argument("--val-freq", type=int, default=None)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    base = cfg["experiment_id"]

    merged = load_metadata_and_splits()
    groups = DATASET_GROUPS[cfg["dataset"]]
    df = merged if groups is None else merged[merged["Storage Group"].isin(groups)].reset_index(drop=True)
    n_samples = df[["Storage Group", "Sample"]].drop_duplicates().shape[0]
    print(f"[cv] {args.folds}-fold on '{cfg['dataset']}': {len(df)} images / {n_samples} samples "
          f"(split at the group level, §2.1) model={cfg['model']}")

    fold_metrics = []
    for f, (tr, va) in enumerate(group_kfold(df, args.folds, args.seed)):
        n_tr_s = tr[["Storage Group", "Sample"]].drop_duplicates().shape[0]
        n_va_s = va[["Storage Group", "Sample"]].drop_duplicates().shape[0]
        print(f"\n===== fold {f + 1}/{args.folds}  "
              f"train {len(tr)} imgs/{n_tr_s} samples · val {len(va)} imgs/{n_va_s} samples =====")
        rec = train(cfg, smoke=args.smoke_test, epochs_override=args.epochs,
                    val_freq_override=args.val_freq, train_df=tr, val_df=va,
                    exp_id=f"{base}_fold{f}")
        m = rec["best_metrics"]
        fold_metrics.append(m)
        print(f"[cv] fold {f}: acc={m['exact_accuracy']:.3f} within1={m['within_1_stage_accuracy']:.3f} "
              f"MAE={m['stage_mae']:.3f} QWK={m['qwk']:.3f}")

    agg = {k: {"mean": float(np.mean([m[k] for m in fold_metrics])),
               "std": float(np.std([m[k] for m in fold_metrics], ddof=0)),
               "per_fold": [float(m[k]) for m in fold_metrics]} for k in AGG_KEYS}
    agg["per_class_recall_mean"] = [float(x) for x in
                                    np.array([m["per_class_recall"] for m in fold_metrics]).mean(0)]
    summary = {"model": cfg["model"], "dataset": cfg["dataset"], "folds": args.folds,
               "seed": args.seed, "n_samples": int(n_samples), "smoke_test": args.smoke_test,
               "aggregate": agg}
    out = OUTPUT_DIR / "cv" / base / f"cv_{cfg['dataset']}_{args.folds}fold.json"
    save_json(summary, out)

    print(f"\n=== {args.folds}-fold CV summary — {cfg['model']} on '{cfg['dataset']}' (mean ± std) ===")
    for k in AGG_KEYS:
        print(f"  {k:26s} {agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}")
    print(f"[cv] saved -> {out}")


if __name__ == "__main__":
    main()
