"""K-fold cross-validation for the classifier (group-aware) — a more robust estimate than the
single fixed split.

Folds are built on the (Storage Group, Sample) key, NEVER image-level (CLAUDE.md §2.1), so the
same avocado never lands in both train and val within a fold. Trains the config's model once per
fold, records the best-val metric bundle per fold, and reports mean ± std across folds. Uses the
same curated data (metadata_clean.csv / manifest) and metrics (metrics.image_level_metrics) as the
single-split runs, so the numbers are comparable.

Resumable. Each fold's result is saved to cv/<exp>/fold<f>.json as it completes, and a rerun skips
folds whose result already exists — so an interrupted CV picks up where it stopped (important on
CPU where a fold can take many hours). --recover-folds "0,1" additionally re-evaluates an existing
best.pt for those folds instead of retraining them (used to reclaim folds already trained by a
cancelled run). Folds are deterministic (seeded), so a recovered fold's val set matches the
original run's exactly.

CV runs over ALL samples of the chosen dataset (the fixed train/val/test Split column is ignored).

Run:
  python -m src.training.cv_train --config configs/paper/general_resnet18.yaml --folds 5
  python -m src.training.cv_train --config ... --folds 5 --recover-folds 0,1   # resume: reuse folds 0,1
"""
from __future__ import annotations

import argparse
import json

import numpy as np

try:
    from ..common.metrics import image_level_metrics  # noqa: F401  (kept for parity/imports)
    from ..common.models import build_model
    from ..common.transforms import evaluation_transform
    from ..common.utils import OUTPUT_DIR, get_device, load_config, save_json, set_seed
    from ..data.dataset import DATASET_GROUPS, load_metadata_and_splits, make_loader
    from .train import evaluate_val, train
except ImportError:  # src/ on sys.path (tests, serving): `..` would escape the top-level package
    from common.metrics import image_level_metrics  # noqa: F401
    from common.models import build_model
    from common.transforms import evaluation_transform
    from common.utils import OUTPUT_DIR, get_device, load_config, save_json, set_seed
    from data.dataset import DATASET_GROUPS, load_metadata_and_splits, make_loader
    from training.train import evaluate_val, train

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


def evaluate_fold_checkpoint(cfg: dict, val_df, ckpt_path) -> dict:
    """Re-evaluate an existing best.pt on val_df -> full metric bundle (recover a done fold cheaply,
    no retraining). The seeded folds guarantee val_df matches the run that produced the checkpoint."""
    import torch
    import torch.nn as nn

    device = get_device()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg["model"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    loader = make_loader(val_df, evaluation_transform(cfg.get("img_size", 224)),
                         cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 4))
    _, _, m = evaluate_val(model, loader, device, nn.CrossEntropyLoss())
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Group-aware k-fold CV for the classifier (resumable)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs (per fold)")
    ap.add_argument("--val-freq", type=int, default=None)
    ap.add_argument("--recover-folds", default="",
                    help="comma fold indices to recover from an existing best.pt instead of training, e.g. 0,1")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    base = cfg["experiment_id"]
    recover = {int(x) for x in args.recover_folds.split(",") if x.strip() != ""}

    merged = load_metadata_and_splits()
    groups = DATASET_GROUPS[cfg["dataset"]]
    df = merged if groups is None else merged[merged["Storage Group"].isin(groups)].reset_index(drop=True)
    n_samples = df[["Storage Group", "Sample"]].drop_duplicates().shape[0]
    print(f"[cv] {args.folds}-fold on '{cfg['dataset']}': {len(df)} images / {n_samples} samples "
          f"(group-level, §2.1) model={cfg['model']} recover={sorted(recover) or '-'}")

    outdir = OUTPUT_DIR / "cv" / base
    ckpt_root = OUTPUT_DIR / "checkpoints"
    fold_metrics = []
    for f, (tr, va) in enumerate(group_kfold(df, args.folds, args.seed)):
        fj = outdir / f"fold{f}.json"
        if fj.exists():                                        # resume: already have this fold
            m = json.loads(fj.read_text(encoding="utf-8"))["metrics"]
            print(f"[cv] fold {f}: loaded existing result (resume, skip)")
        else:
            ckpt = ckpt_root / f"{base}_fold{f}" / "best.pt"
            if f in recover and ckpt.exists():                # reclaim a fold trained by an earlier run
                print(f"[cv] fold {f}: RECOVER from checkpoint (no training) -> {ckpt}")
                m = evaluate_fold_checkpoint(cfg, va, ckpt)
            else:                                             # train this fold
                n_tr_s = tr[["Storage Group", "Sample"]].drop_duplicates().shape[0]
                n_va_s = va[["Storage Group", "Sample"]].drop_duplicates().shape[0]
                print(f"\n===== fold {f + 1}/{args.folds}  "
                      f"train {len(tr)} imgs/{n_tr_s} samples · val {len(va)} imgs/{n_va_s} samples =====")
                rec = train(cfg, smoke=args.smoke_test, epochs_override=args.epochs,
                            val_freq_override=args.val_freq, train_df=tr, val_df=va,
                            exp_id=f"{base}_fold{f}")
                m = rec["best_metrics"]
            save_json({"fold": f, "metrics": m}, fj)          # persist so a later rerun skips it
        fold_metrics.append(m)
        print(f"[cv] fold {f}: acc={m['exact_accuracy']:.3f} within1={m['within_1_stage_accuracy']:.3f} "
              f"MAE={m['stage_mae']:.3f} QWK={m['qwk']:.3f}")

    agg = {k: {"mean": float(np.mean([m[k] for m in fold_metrics])),
               "std": float(np.std([m[k] for m in fold_metrics], ddof=0)),
               "per_fold": [float(m[k]) for m in fold_metrics]} for k in AGG_KEYS}
    agg["per_class_recall_mean"] = [float(x) for x in
                                    np.array([m["per_class_recall"] for m in fold_metrics]).mean(0)]
    summary = {"model": cfg["model"], "dataset": cfg["dataset"], "folds": args.folds,
               "seed": args.seed, "n_samples": int(n_samples), "recovered_folds": sorted(recover),
               "smoke_test": args.smoke_test, "aggregate": agg}
    summ_path = outdir / f"cv_{cfg['dataset']}_{args.folds}fold.json"
    save_json(summary, summ_path)

    print(f"\n=== {args.folds}-fold CV summary — {cfg['model']} on '{cfg['dataset']}' (mean ± std) ===")
    for k in AGG_KEYS:
        print(f"  {k:26s} {agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}")
    print(f"[cv] saved -> {summ_path}")


if __name__ == "__main__":
    main()
