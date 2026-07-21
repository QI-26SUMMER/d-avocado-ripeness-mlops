"""GMM color baseline (classical ML) — a comparison point for the ResNet classifier.

Idea (as proposed): represent each image by its avocado's mean RGB — a point in color space.
Fit one Gaussian mixture per ripeness stage on TRAIN; classify a new image to the stage whose
mixture gives the highest posterior ("nearest color distribution"). Same split (splits.csv),
same curated data (metadata_clean.csv, the manifest), and same metrics (metrics.image_level_metrics)
as ResNet, so the two numbers are directly comparable (CLAUDE.md §4).

Run (after `python -m src.validate_data` and `python -m src.split`, with the images on disk):
  python -m src.gmm_baseline                          # general, 2 components/class
  python -m src.gmm_baseline --dataset general --components 3

Deliberate choices (see CLAUDE.md):
  - Fit on TRAIN, evaluate on TEST — fitting on test would leak (§2.1/§4).
  - Feature = mean RGB of AVOCADO pixels only. The light-box background is white and would
    dominate a full-frame mean, so near-white pixels are dropped. This threshold is valid ONLY
    because the train/eval data is clean light-box; a real user photo must NOT be masked by a
    white threshold (§3) — it would go through the SAM/rembg crop (src/preprocess.py) instead.
  - This baseline reports SINGLE-IMAGE metrics. Best-side reporting (§4) can be layered on later
    to match evaluate.py; single-image is the primary head-to-head number.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

try:
    from .data import IMAGE_DIR
    from .metrics import image_level_metrics, summarize
    from .utils import OUTPUT_DIR, save_json, set_seed
except ImportError:
    from data import IMAGE_DIR
    from metrics import image_level_metrics, summarize
    from utils import OUTPUT_DIR, save_json, set_seed

if TYPE_CHECKING:
    import pandas as pd

METADATA_CLEAN = OUTPUT_DIR / "splits" / "metadata_clean.csv"
SPLITS_CSV = OUTPUT_DIR / "splits" / "splits.csv"
STAGES = [1, 2, 3, 4, 5]
WHITE_THRESH = 235   # a pixel is 'background' when all 3 channels >= this (near-white light box)
DATASET_GROUPS = {"general": None, "T10": ["T10"], "T20": ["T20"], "Tam": ["Tam"], "Tamb": ["Tam"]}


# ─── Feature: one image → its avocado's mean RGB (a 3D point) ────────────────
def avocado_color(img, white_thresh: int = WHITE_THRESH, resize: int = 128) -> np.ndarray:
    """Mean RGB of the avocado (non-near-white) pixels. Returns (3,) float in [0, 255].

    Downscaled first for speed. If almost every pixel is near-white (defensive; shouldn't happen
    on a real avocado photo) it falls back to the full-frame mean.
    """
    img = img.convert("RGB")
    if resize:
        img = img.resize((resize, resize), Image.BILINEAR)
    arr = np.asarray(img).reshape(-1, 3).astype(np.float32)
    fg = arr[(arr < white_thresh).any(axis=1)]        # keep pixels that are NOT near-white
    return (fg if len(fg) >= 0.02 * len(arr) else arr).mean(axis=0)


def extract_features(df: "pd.DataFrame", image_dir: Path = IMAGE_DIR) -> np.ndarray:
    """(N, 3) mean-RGB features for every row's image."""
    feats = np.empty((len(df), 3), np.float32)
    for i, fn in enumerate(df["File Name"].astype(str)):
        feats[i] = avocado_color(Image.open(Path(image_dir) / f"{fn}.jpg"))
    return feats


# ─── Model: one GMM per stage, classify by max posterior (MAP) ──────────────
class GMMColorClassifier:
    """Fit a GaussianMixture per ripeness stage; predict the stage maximizing
    (class log-likelihood + log prior). That MAP rule assigns each point to the stage whose
    color distribution it is 'closest' to — the requested nearest-cluster behaviour, done as a
    proper generative Bayes classifier."""

    def __init__(self, n_components: int = 2, seed: int = 42):
        self.n_components = n_components
        self.seed = seed

    def fit(self, X, y):
        from sklearn.mixture import GaussianMixture

        X = np.asarray(X, float)
        y = np.asarray(y, int)
        self.gmms_, self.log_prior_ = {}, {}
        for c in STAGES:
            Xc = X[y == c]
            k = max(1, min(self.n_components, len(Xc)))
            g = GaussianMixture(n_components=k, covariance_type="full",
                                random_state=self.seed, reg_covar=1e-3)
            g.fit(Xc)
            self.gmms_[c] = g
            self.log_prior_[c] = float(np.log(len(Xc) / len(X)))
        return self

    def log_posterior(self, X) -> np.ndarray:
        """(N, 5) — per-stage log-likelihood + log prior (unnormalised log posterior)."""
        X = np.asarray(X, float)
        return np.column_stack(
            [self.gmms_[c].score_samples(X) + self.log_prior_[c] for c in STAGES])

    def predict(self, X) -> np.ndarray:
        return np.array(STAGES)[self.log_posterior(X).argmax(axis=1)]


# ─── Data: reuse the exact ResNet split (no torch dependency here) ──────────
def load_split_frames(dataset: str = "general"):
    """(train_df, test_df) from metadata_clean + splits for the given dataset — the SAME split
    ResNet uses. Replicated (not importing dataset.py) so this baseline needs no torch."""
    import pandas as pd

    if not METADATA_CLEAN.exists() or not SPLITS_CSV.exists():
        raise SystemExit("Run `python -m src.validate_data` and `python -m src.split` first "
                         f"({METADATA_CLEAN} / {SPLITS_CSV} not found).")
    if dataset not in DATASET_GROUPS:
        raise SystemExit(f"unknown dataset {dataset!r} (expected one of {list(DATASET_GROUPS)})")
    meta = pd.read_csv(METADATA_CLEAN)
    splits = pd.read_csv(SPLITS_CSV)
    m = meta.merge(splits[["Storage Group", "Sample", "Split"]],
                   on=["Storage Group", "Sample"], how="left")
    assert m["Split"].notna().all(), "some images have no split assigned"
    groups = DATASET_GROUPS[dataset]
    if groups is not None:
        m = m[m["Storage Group"].isin(groups)]
    tr = m[m["Split"] == "train"].reset_index(drop=True)
    te = m[m["Split"] == "test"].reset_index(drop=True)
    return tr, te


def scatter_rgb(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """3D scatter of avocado mean-RGB colored by stage — the literal 'plot the RGB as points'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_color = {1: "#4c9a2a", 2: "#9aa53c", 3: "#c08a2e", 4: "#7a4a2b", 5: "#39271f"}
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    for c in STAGES:
        P = X[y == c]
        if len(P):
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=6, alpha=0.35,
                       color=stage_color[c], label=f"stage {c}")
    ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
    ax.set_title("Avocado mean RGB by ripeness stage (train)")
    ax.legend(markerscale=2.5, loc="upper left", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="GMM RGB color baseline vs ResNet")
    ap.add_argument("--dataset", default="general", choices=list(DATASET_GROUPS))
    ap.add_argument("--components", type=int, default=2, help="Gaussian components per stage")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    tr, te = load_split_frames(args.dataset)
    print(f"[gmm] extracting mean-RGB features: train {len(tr)} / test {len(te)} images ...")
    Xtr, ytr = extract_features(tr), tr["label"].to_numpy(int)
    Xte, yte = extract_features(te), te["label"].to_numpy(int)

    clf = GMMColorClassifier(args.components, args.seed).fit(Xtr, ytr)
    m = image_level_metrics(yte, clf.predict(Xte))
    print(f"[gmm color baseline | {args.dataset} | k={args.components}] single-image: {summarize(m)}")

    outdir = OUTPUT_DIR / "gmm_baseline"
    save_json({"model": "gmm_rgb_color", "dataset": args.dataset,
               "n_components": args.components, "seed": args.seed,
               "feature": "mean RGB of avocado (non-white) pixels", "metrics": m},
              outdir / f"gmm_{args.dataset}_k{args.components}.json")
    scatter_rgb(Xtr, ytr, outdir / f"gmm_{args.dataset}_rgb_scatter.png")
    print(f"[gmm] saved metrics + RGB scatter -> {outdir}")


if __name__ == "__main__":
    main()
