"""Training (config-driven) — paper reproduction §10·§11.

Usage:
  python -m src.train --config configs/paper/general_resnet18.yaml --smoke-test
  python -m src.train --config configs/paper/general_resnet18.yaml

Paper-published settings: epochs>=30, batch 128, ResNet-18 lr 0.01 (10ep×0.1), AlexNet lr
  0.001 (10ep×0.1), validation every 10 iters, patience 150 checks, saves best val
  accuracy/loss model, oversampling on train only.
Not published in the paper (→ implementation choice, specified in config): optimizer/
  momentum/weight_decay, split seed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from .dataset import load_metadata_and_splits, make_loader, subset_frame
    from .metrics import image_level_metrics, summarize
    from .models import build_model
    from .sampler import effective_class_counts, paper_oversample_indices
    from .transforms import build_train_transform, evaluation_transform
    from .utils import OUTPUT_DIR, get_device, git_commit_hash, load_config, save_json, set_seed
except ImportError:
    from dataset import load_metadata_and_splits, make_loader, subset_frame
    from metrics import image_level_metrics, summarize
    from models import build_model
    from sampler import effective_class_counts, paper_oversample_indices
    from transforms import build_train_transform, evaluation_transform
    from utils import OUTPUT_DIR, get_device, git_commit_hash, load_config, save_json, set_seed


def build_optimizer(model, cfg):
    """Not published in the paper → implementation choice. Defaults to SGD (momentum=0.9, weight_decay=1e-4)."""
    o = cfg.get("optimizer", {})
    name = o.get("name", "sgd").lower()
    lr = cfg["lr"]
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=o.get("momentum", 0.9),
                               weight_decay=o.get("weight_decay", 1e-4))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=o.get("weight_decay", 0.0))
    raise ValueError(f"unknown optimizer {name!r}")


@torch.no_grad()
def evaluate_val(model, loader, device, crit) -> tuple[float, float, dict]:
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += crit(out, y).item() * len(y)
        n += len(y)
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.cpu().numpy())
    y_true = np.concatenate(ys) + 1
    y_pred = np.concatenate(ps) + 1
    m = image_level_metrics(y_true, y_pred)
    return m["exact_accuracy"], loss_sum / n, m


def train(cfg: dict, smoke: bool = False,
          epochs_override: int | None = None, val_freq_override: int | None = None) -> dict:
    set_seed(cfg.get("seed", 42), deterministic=cfg.get("deterministic", True))
    device = get_device()
    exp_id = cfg["experiment_id"]
    ckpt_dir = OUTPUT_DIR / "checkpoints" / exp_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    merged = load_metadata_and_splits()
    train_df = subset_frame(merged, cfg["dataset"], "train")
    val_df = subset_frame(merged, cfg["dataset"], "val")

    img_size = cfg.get("img_size", 224)
    num_workers = 0 if smoke else cfg.get("num_workers", 4)
    if smoke:  # tiny subset only
        train_df = train_df.iloc[: cfg["batch_size"] * 2].reset_index(drop=True)
        val_df = val_df.iloc[: cfg["batch_size"]].reset_index(drop=True)

    # oversampling (train only, §7)
    balance = cfg.get("balance", "paper_oversample")
    if balance == "paper_oversample":
        idx = paper_oversample_indices(train_df["label"].values, seed=cfg.get("seed", 42))
        eff = effective_class_counts(train_df["label"].values, idx)
        print(f"[oversample] effective train class counts (original→balanced): {eff}")
    elif balance == "none":
        idx = None
    else:
        raise ValueError(f"unknown balance {balance!r}")

    train_tf = build_train_transform(cfg.get("augmentation", "paper"), img_size)
    eval_tf = evaluation_transform(img_size)
    g = torch.Generator().manual_seed(cfg.get("seed", 42))
    train_loader = make_loader(train_df, train_tf, cfg["batch_size"], shuffle=(idx is None),
                               num_workers=num_workers, sampler_indices=idx, generator=g)
    val_loader = make_loader(val_df, eval_tf, cfg["batch_size"], shuffle=False,
                             num_workers=num_workers)

    model = build_model(cfg["model"], pretrained=cfg.get("pretrained", True)).to(device)
    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.get("lr_step", 10),
                                            gamma=cfg.get("lr_gamma", 0.1))
    crit = nn.CrossEntropyLoss()

    epochs = 1 if smoke else (epochs_override or cfg["epochs"])
    # Paper: validation every 10 iters. For actual GPU runs, this can be relaxed via
    # val_freq_override (e.g. once per epoch) for wall-clock reasons (a practical
    # implementation choice, noted in the report). Uses the paper's value if no override is given.
    val_freq = 2 if smoke else (val_freq_override or cfg.get("validation", {}).get("frequency_iters", 10))
    patience = 3 if smoke else cfg.get("validation", {}).get("patience_checks", 150)

    print(f"[{exp_id}] dataset={cfg['dataset']} model={cfg['model']} aug={cfg.get('augmentation')} "
          f"balance={balance} device={device}")
    print(f"  train imgs={len(train_df)}(over→{len(idx) if idx else len(train_df)}) "
          f"val imgs={len(val_df)} batch={cfg['batch_size']} lr={cfg['lr']} epochs={epochs}")

    history, best = [], {"val_acc": -1.0, "val_loss": float("inf"), "iter": -1, "epoch": -1}
    checks_no_improve, git_hash = 0, git_commit_hash()
    g_iter, stop = 0, False
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            g_iter += 1
            if g_iter % val_freq == 0:
                va, vl, _ = evaluate_val(model, val_loader, device, crit)
                history.append({"iter": g_iter, "epoch": epoch, "train_loss": float(loss.item()),
                                "val_acc": va, "val_loss": vl, "lr": opt.param_groups[0]["lr"]})
                improved = (va > best["val_acc"]) or (va == best["val_acc"] and vl < best["val_loss"])
                if improved:
                    best = {"val_acc": va, "val_loss": vl, "iter": g_iter, "epoch": epoch}
                    torch.save({"model": model.state_dict(), "cfg": cfg, "best": best},
                               ckpt_dir / "best.pt")
                    checks_no_improve = 0
                else:
                    checks_no_improve += 1
                model.train()
                if checks_no_improve >= patience:
                    print(f"  early stop: no improvement for {patience} val checks (iter {g_iter})")
                    stop = True
                    break
        sched.step()
        if history:
            print(f"  e{epoch} iter{g_iter} {time.time()-t0:4.0f}s "
                  f"val_acc={history[-1]['val_acc']:.3f} val_loss={history[-1]['val_loss']:.3f} "
                  f"lr={opt.param_groups[0]['lr']:.4g} (best acc={best['val_acc']:.3f}@e{best['epoch']})")
        if stop:
            break

    # Save record (§11)
    record = {
        "experiment_id": exp_id, "config": cfg, "git_commit": git_hash,
        "device": device, "pretrained": cfg.get("pretrained", True),
        "optimizer": cfg.get("optimizer", {"name": "sgd", "momentum": 0.9, "weight_decay": 1e-4}),
        "scheduler": {"type": "StepLR", "step": cfg.get("lr_step", 10), "gamma": cfg.get("lr_gamma", 0.1)},
        "epochs_run": epoch, "best": best, "history": history,
        "checkpoint": str(ckpt_dir / "best.pt"),
        "note_impl_choices": "optimizer/momentum/weight_decay/split_seed are not published in the paper (implementation choice)",
        "smoke_test": smoke,
    }
    save_json(record, ckpt_dir / "history.json")
    save_json(cfg, ckpt_dir / "config.json")
    print(f"  best val_acc={best['val_acc']:.3f} (loss={best['val_loss']:.3f}) → {ckpt_dir/'best.pt'}")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="Override config epochs (practical runs)")
    ap.add_argument("--val-freq", type=int, default=None, help="Override validation frequency_iters")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg, smoke=args.smoke_test, epochs_override=args.epochs, val_freq_override=args.val_freq)


if __name__ == "__main__":
    main()
