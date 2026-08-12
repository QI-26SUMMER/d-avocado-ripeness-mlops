"""Reproducibility / config / logging utilities (CLAUDE.md §8, paper reproduction §11).

For settings not disclosed in the paper, we provide a default here but always allow it
to be overridden via config.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix the Python/NumPy/torch (CPU/CUDA) seed. §11 reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """Fix the DataLoader worker seed (§11), based on torch.initial_seed."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(path: str | Path) -> dict:
    """Load a YAML config. Requires pyyaml."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_commit_hash() -> str:
    """Current commit hash (§11 experiment logging). Returns 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# Project paths. parents[2] because this file is src/common/utils.py — two levels below the root.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Default is local outputs. When running on GCP (Vertex), point AVOCADO_OUTPUT_DIR at the
# GCS mount (/gcs/...) so checkpoints/history are persisted durably (no effect on local runs).
OUTPUT_DIR = Path(os.environ.get("AVOCADO_OUTPUT_DIR", str(REPO_ROOT / "outputs" / "paper_reproduction")))
