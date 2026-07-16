"""Sequentially train the 8 paper-reproduction experiments, then evaluate all (paper reproduction §9·§16).

Run: python scripts/run_paper_experiments.py
Options:
  --configs-dir configs/paper   folder of experiment configs
  --val-freq 80                 override for validation frequency_iters (practical run)
  --epochs N                    override for epochs
  --smoke                       run each experiment as smoke only (pipeline check)

Trains General×ResNet-18 first → then the rest, and finally runs evaluate --all.
Each experiment has an independent checkpoint (outputs/paper_reproduction/checkpoints/<experiment_id>/).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# General×ResNet-18 first (smoke/validation priority), then the rest
ORDER = ["general_resnet18", "general_alexnet",
         "t10_resnet18", "t10_alexnet",
         "t20_resnet18", "t20_alexnet",
         "tamb_resnet18", "tamb_alexnet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default="configs/paper")
    ap.add_argument("--val-freq", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cdir = ROOT / args.configs_dir
    configs = [cdir / f"{name}.yaml" for name in ORDER if (cdir / f"{name}.yaml").exists()]
    print(f"{len(configs)} experiments: {[c.stem for c in configs]}")

    for c in configs:
        cmd = [sys.executable, "-m", "src.train", "--config", str(c)]
        if args.smoke:
            cmd.append("--smoke-test")
        else:
            cmd += ["--val-freq", str(args.val_freq)]
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]
        print(f"\n$ {' '.join(cmd)}")
        subprocess.run(cmd, cwd=ROOT, check=True)

    print("\n=== Evaluate all ===")
    subprocess.run([sys.executable, "-m", "src.evaluate", "--all"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
