"""Evaluation — image-level / best-side / two-side + shelf-life (paper reproduction §12·§13).

Usage:
  python -m src.training.evaluate --config configs/paper/general_resnet18.yaml   # single experiment
  python -m src.training.evaluate --all                                          # all experiments

Clearly separates three evaluation modes (§12.2):
  A. Image-level          : each image treated independently
  B. Best-side (oracle)   : correct if either side (same sample, same day) is correct
                            stage error = min(|pred_a−y|, |pred_b−y|)  — not deployable, for paper comparison only
  C. Two-side deployable  : average softmax of both sides → argmax (usable in an actual service)

Shelf-life (§13): plugs the CNN's 5-stage prediction into the per-storage-group linear
formula (no separate regression model).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from ..common.metrics import image_level_metrics
    from ..common.models import build_model
    from ..common.shelf_life import ALPHA_5STAGE, derive_actual_days_left, estimate_days_left
    from ..common.transforms import evaluation_transform
    from ..common.utils import OUTPUT_DIR, get_device, load_config, save_json
    from ..data.dataset import AvocadoImageDataset, load_metadata_and_splits, subset_frame
except ImportError:  # src/ on sys.path (tests, serving): `..` would escape the top-level package
    from common.metrics import image_level_metrics
    from common.models import build_model
    from common.shelf_life import ALPHA_5STAGE, derive_actual_days_left, estimate_days_left
    from common.transforms import evaluation_transform
    from common.utils import OUTPUT_DIR, get_device, load_config, save_json
    from data.dataset import AvocadoImageDataset, load_metadata_and_splits, subset_frame

OBS_KEYS = ["Storage Group", "Sample", "Day of Experiment"]


@torch.no_grad()
def infer_test(cfg, model, device) -> pd.DataFrame:
    """Softmax + pred for every image in the test set. Preserves test_df row order."""
    merged = load_metadata_and_splits()
    test_df = subset_frame(merged, cfg["dataset"], "test").copy()
    ds = AvocadoImageDataset(test_df, transform=evaluation_transform(cfg.get("img_size", 224)))
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 128), shuffle=False, num_workers=0)
    probs = []
    model.eval()
    for x, _ in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    probs = np.concatenate(probs)
    test_df["pred"] = probs.argmax(1) + 1
    for k in range(5):
        test_df[f"p{k+1}"] = probs[:, k]
    # actual days_left for shelf-life (end_day is computable since all observations of the test samples are in the test set)
    dl = derive_actual_days_left(test_df, censored="drop", clip_negative_to_zero=True)
    test_df["actual_days_left"] = dl["actual_days_left"].values
    test_df["is_censored"] = dl["is_censored"].values
    return test_df


def _shelf_loss(index_series, group_series, actual_series) -> dict:
    m = actual_series.notna().values
    if m.sum() == 0:
        return {"mean_loss_days": None, "std_days": None, "n": 0}
    est = np.array([estimate_days_left(i, g)
                    for i, g in zip(np.asarray(index_series)[m], np.asarray(group_series)[m])])
    loss = np.abs(est - np.asarray(actual_series)[m].astype(float))
    return {"mean_loss_days": float(loss.mean()), "std_days": float(loss.std(ddof=0)),
            "n": int(m.sum())}


def evaluate_experiment(cfg, model, device) -> dict:
    test = infer_test(cfg, model, device)

    # A. image-level
    A = image_level_metrics(test["label"].values, test["pred"].values)

    # Build observation-level (sample×day) rows — best-side / two-side
    rows = []
    for keys, g in test.groupby(OBS_KEYS):
        y = int(g["label"].iloc[0])  # a/b share the same label (verified). Take the first value defensively.
        preds = g["pred"].values
        prob_mean = g[[f"p{k}" for k in range(1, 6)]].mean().values
        two_side = int(prob_mean.argmax() + 1)
        best_err = int(np.min(np.abs(preds - y)))
        rows.append({
            "Storage Group": keys[0], "Sample": keys[1], "Day of Experiment": keys[2],
            "y": y, "two_side_pred": two_side,
            "best_side_correct": int((preds == y).any()),
            "best_side_stage_err": best_err,
            # the 'stage' chosen by best-side (the prediction of the side with minimum loss)
            "best_side_pred": int(preds[np.argmin(np.abs(preds - y))]),
            "actual_days_left": g["actual_days_left"].iloc[0],
        })
    obs = pd.DataFrame(rows)

    # B. best-side (oracle)
    B = {
        "exact_accuracy": float(obs["best_side_correct"].mean()),
        "stage_mae": float(obs["best_side_stage_err"].mean()),
        "within_1_stage_accuracy": float((obs["best_side_stage_err"] <= 1).mean()),
        "n_observations": int(len(obs)),
    }
    # C. two-side deployable
    C = image_level_metrics(obs["y"].values, obs["two_side_pred"].values)
    C["n_observations"] = int(len(obs))

    # 4 shelf-life variants (§13)
    shelf = {
        "attributed_by_picture": _shelf_loss(test["label"], test["Storage Group"], test["actual_days_left"]),
        "predicted_by_picture": _shelf_loss(test["pred"], test["Storage Group"], test["actual_days_left"]),
        "predicted_by_best_side_sample": _shelf_loss(obs["best_side_pred"], obs["Storage Group"], obs["actual_days_left"]),
        "predicted_by_prob_avg_sample": _shelf_loss(obs["two_side_pred"], obs["Storage Group"], obs["actual_days_left"]),
        "by_group": {},
        "note": "actual_days_left is the number of days until first reaching stage 5. Censored samples excluded (drop), values past stage 5 clipped to 0.",
        "n_censored_samples_in_test": int(test.loc[test["is_censored"], "Sample"].nunique()),
    }
    for grp, gg in test.groupby("Storage Group"):
        oo = obs[obs["Storage Group"] == grp]
        shelf["by_group"][grp] = {
            "alpha": ALPHA_5STAGE[grp],
            "attributed_by_picture": _shelf_loss(gg["label"], gg["Storage Group"], gg["actual_days_left"]),
            "predicted_by_picture": _shelf_loss(gg["pred"], gg["Storage Group"], gg["actual_days_left"]),
            "predicted_by_prob_avg_sample": _shelf_loss(oo["two_side_pred"], oo["Storage Group"], oo["actual_days_left"]),
        }

    return {"experiment_id": cfg["experiment_id"], "dataset": cfg["dataset"], "model": cfg["model"],
            "A_image_level": A, "B_best_side_oracle": B, "C_two_side_deployable": C,
            "shelf_life": shelf, "_predictions": test, "_observations": obs}


def _run_one(config_path: str, device: str):
    cfg = load_config(config_path)
    ckpt_path = OUTPUT_DIR / "checkpoints" / cfg["experiment_id"] / "best.pt"
    if not ckpt_path.exists():
        print(f"  [skip] checkpoint not found: {ckpt_path} (train first)")
        return None
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg["model"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    res = evaluate_experiment(cfg, model, device)

    eid = cfg["experiment_id"]
    # unlike save_json, to_csv does not auto-create the parent folder → create it explicitly
    # (pandas raises OSError if OUTPUT_DIR/predictions doesn't exist yet). save_json does mkdir internally.
    pred_dir = OUTPUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    res["_predictions"].to_csv(pred_dir / f"{eid}_test_predictions.csv", index=False)
    res["_observations"].to_csv(pred_dir / f"{eid}_test_observations.csv", index=False)
    save_json(res["A_image_level"]["confusion_matrix"],
              OUTPUT_DIR / "confusion_matrices" / f"{eid}_confusion.json")
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    save_json(out, OUTPUT_DIR / "metrics" / f"{eid}_metrics.json")
    A, B, C = res["A_image_level"], res["B_best_side_oracle"], res["C_two_side_deployable"]
    print(f"[{eid}]")
    print(f"  A image-level : acc={A['exact_accuracy']:.3f} within1={A['within_1_stage_accuracy']:.3f} "
          f"MAE={A['stage_mae']:.3f} macroF1={A['macro_f1']:.3f} QWK={A['qwk']:.3f}")
    print(f"  B best-side   : acc={B['exact_accuracy']:.3f} within1={B['within_1_stage_accuracy']:.3f} MAE={B['stage_mae']:.3f} (oracle)")
    print(f"  C two-side    : acc={C['exact_accuracy']:.3f} within1={C['within_1_stage_accuracy']:.3f} MAE={C['stage_mae']:.3f}")
    sl = res["shelf_life"]
    print(f"  shelf-life loss (days): attributed={sl['attributed_by_picture']['mean_loss_days']:.3f} "
          f"predicted(pic)={sl['predicted_by_picture']['mean_loss_days']:.3f} "
          f"best-side={sl['predicted_by_best_side_sample']['mean_loss_days']:.3f} "
          f"prob-avg={sl['predicted_by_prob_avg_sample']['mean_loss_days']:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--experiment-dir", default=None, help="(compat) ignored — evaluates all of configs/paper")
    args = ap.parse_args()
    device = get_device()

    if args.all or args.experiment_dir or not args.config:
        cfgs = sorted(Path("configs/paper").glob("*.yaml"))
        for c in cfgs:
            _run_one(str(c), device)
    else:
        _run_one(args.config, device)


if __name__ == "__main__":
    main()
