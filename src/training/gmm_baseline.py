"""GMM color baseline (classical ML) — a comparison point for the ResNet classifier.

Idea: represent each image by its avocado's COLOR statistics — a point in color space. Fit one
Gaussian mixture per ripeness stage on TRAIN; classify a new image to the stage whose mixture
gives the highest posterior ("nearest color distribution"). Same split (splits.csv), same curated
data (metadata_clean.csv, the manifest), same metrics (metrics.image_level_metrics) as ResNet, so
the numbers are directly comparable (CLAUDE.md §4).

Two feature sets, run side by side to isolate what richer color features buy:
  - "rgb"  : mean RGB (3-D). The original literal 'plot the RGB as points' baseline.
  - "rich" : mean RGB + std RGB + mean HSV + mean Lab (12-D). std adds within-fruit variation
             (blemishes/texture darkening), HSV/Lab add perceptual hue & green-red / blue-yellow
             axes — aimed at the stages 2-5 that overlap on the RGB darkening line.
Features are z-scored (StandardScaler) before the GMM. (A full-covariance GMM is invariant to an
affine rescale, so scaling only affects the non-affine 'rich' features / conditioning, not 'rgb'.)

Run (after `python -m src.data.validate_data` and `python -m src.data.split`, with images on disk):
  python -m src.training.gmm_baseline                          # general, 2 components/class, runs rgb + rich
  python -m src.training.gmm_baseline --dataset general --components 3

Deliberate choices (see CLAUDE.md):
  - Fit on TRAIN, evaluate on TEST — fitting on test would leak (§2.1/§4).
  - Feature from AVOCADO pixels only; near-white light-box background is dropped. Valid ONLY on the
    clean light-box data — a real user photo must NOT be masked by a white threshold (§3); it goes
    through the SAM/rembg crop instead.
  - Single-image metrics (best-side, §4, can be layered on later).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

try:
    from ..common.metrics import image_level_metrics, summarize
    from ..common.utils import OUTPUT_DIR, save_json, set_seed
    from ..data.data import IMAGE_DIR
except ImportError:  # src/ on sys.path (tests, serving): `..` would escape the top-level package
    from common.metrics import image_level_metrics, summarize
    from common.utils import OUTPUT_DIR, save_json, set_seed
    from data.data import IMAGE_DIR

if TYPE_CHECKING:
    import pandas as pd

METADATA_CLEAN = OUTPUT_DIR / "splits" / "metadata_clean.csv"
SPLITS_CSV = OUTPUT_DIR / "splits" / "splits.csv"
STAGES = [1, 2, 3, 4, 5]
WHITE_THRESH = 235
DATASET_GROUPS = {"general": None, "T10": ["T10"], "T20": ["T20"], "Tam": ["Tam"], "Tamb": ["Tam"]}

# Per-image color stats are always computed as a 12-vector; feature sets select columns of it.
#   [0:3] mean RGB · [3:6] std RGB · [6:9] mean HSV · [9:12] mean Lab
STAT_NAMES = ["meanR", "meanG", "meanB", "stdR", "stdG", "stdB",
              "meanH", "meanS", "meanV", "meanL", "meana", "meanb"]
FEATURE_SETS = {"rgb": slice(0, 3), "rich": slice(0, 12)}


# ─── Color conversion + per-image stats ─────────────────────────────────────
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (…,3) in [0,255] → CIE Lab (D65). Vectorised, no skimage/cv2 dependency."""
    r = np.asarray(rgb, np.float32) / 255.0
    lin = np.where(r <= 0.04045, r / 12.92, ((r + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], np.float32)
    xyz = (lin @ m.T) / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def image_color_stats(img, white_thresh: int = WHITE_THRESH, resize: int = 128) -> np.ndarray:
    """12-vector [meanRGB, stdRGB, meanHSV, meanLab] over the avocado (non-near-white) pixels.

    HSV comes from PIL (0-255); the hue mean is a plain mean, fine here because avocado hues
    (green→brown) don't straddle the red 0/255 wrap. Falls back to all pixels if almost all are
    near-white (defensive)."""
    img = img.convert("RGB")
    if resize:
        img = img.resize((resize, resize), Image.BILINEAR)
    rgb = np.asarray(img).reshape(-1, 3).astype(np.float32)
    hsv = np.asarray(img.convert("HSV")).reshape(-1, 3).astype(np.float32)
    mask = (rgb < white_thresh).any(axis=1)
    if mask.sum() < 0.02 * len(rgb):
        mask = np.ones(len(rgb), bool)
    px, ph = rgb[mask], hsv[mask]
    return np.concatenate([px.mean(0), px.std(0), ph.mean(0), rgb_to_lab(px).mean(0)])


def extract_stats(df: "pd.DataFrame", image_dir: Path = IMAGE_DIR) -> np.ndarray:
    """(N, 12) color-stat features for every row's image (one image pass, all feature sets)."""
    feats = np.empty((len(df), 12), np.float32)
    for i, fn in enumerate(df["File Name"].astype(str)):
        feats[i] = image_color_stats(Image.open(Path(image_dir) / f"{fn}.jpg"))
    return feats


# ─── Model: one GMM per stage (max posterior), on z-scored features ─────────
class GMMColorClassifier:
    """Fit a GaussianMixture per ripeness stage; predict the stage maximizing
    (class log-likelihood + log prior). MAP = assign to the nearest color distribution."""

    def __init__(self, n_components: int = 2, seed: int = 42):
        self.n_components = n_components
        self.seed = seed

    def fit(self, X, y):
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, float)
        y = np.asarray(y, int)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.gmms_, self.log_prior_ = {}, {}
        for c in STAGES:
            Xc = Xs[y == c]
            k = max(1, min(self.n_components, len(Xc)))
            g = GaussianMixture(n_components=k, covariance_type="full",
                                random_state=self.seed, reg_covar=1e-2)
            g.fit(Xc)
            self.gmms_[c] = g
            self.log_prior_[c] = float(np.log(len(Xc) / len(X)))
        return self

    def log_posterior(self, X) -> np.ndarray:
        Xs = self.scaler_.transform(np.asarray(X, float))
        return np.column_stack(
            [self.gmms_[c].score_samples(Xs) + self.log_prior_[c] for c in STAGES])

    def predict(self, X) -> np.ndarray:
        return np.array(STAGES)[self.log_posterior(X).argmax(axis=1)]


# ─── Data: reuse the exact ResNet split (no torch dependency here) ──────────
def load_split_frames(dataset: str = "general"):
    import pandas as pd

    if not METADATA_CLEAN.exists() or not SPLITS_CSV.exists():
        raise SystemExit("Run `python -m src.data.validate_data` and `python -m src.data.split` first "
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


# ─── Visualisation ──────────────────────────────────────────────────────────
STAGE_COLOR = {1: "#4c9a2a", 2: "#9aa53c", 3: "#c08a2e", 4: "#7a4a2b", 5: "#39271f"}


def scatter_rgb(stats: np.ndarray, y: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    for c in STAGES:
        P = stats[y == c]
        if len(P):
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=6, alpha=0.35,
                       color=STAGE_COLOR[c], label=f"stage {c}")
    ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
    ax.set_title("Avocado mean RGB by ripeness stage (train)")
    ax.legend(markerscale=2.5, loc="upper left", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def scatter_lab(stats: np.ndarray, y: np.ndarray, path: Path) -> None:
    """Lab a* (green→red) vs b* (blue→yellow) — the perceptual axes the 'rich' features add."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for c in STAGES:
        P = stats[y == c]
        if len(P):
            ax.scatter(P[:, 10], P[:, 11], s=7, alpha=0.4, color=STAGE_COLOR[c], label=f"stage {c}")
    ax.set_xlabel("Lab a*  (green → red)"); ax.set_ylabel("Lab b*  (blue → yellow)")
    ax.set_title("Avocado mean Lab a*/b* by ripeness stage (train)")
    ax.legend(markerscale=2, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_features_csv(tr, Str, te, Ste, path: Path) -> None:
    """Write per-image color stats (the 'RGB points' + std/HSV/Lab) with metadata + split to CSV."""
    import pandas as pd

    meta_cols = ["File Name", "Storage Group", "Sample", "day", "view", "label", "Split"]

    def frame(df, stats):
        out = df[[c for c in meta_cols if c in df.columns]].copy()
        for j, name in enumerate(STAT_NAMES):
            out[name] = stats[:, j]
        return out

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([frame(tr, Str), frame(te, Ste)], ignore_index=True).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="GMM color baseline (rgb vs rich) vs ResNet")
    ap.add_argument("--dataset", default="general", choices=list(DATASET_GROUPS))
    ap.add_argument("--components", type=int, default=2, help="Gaussian components per stage")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    tr, te = load_split_frames(args.dataset)
    print(f"[gmm] extracting color stats: train {len(tr)} / test {len(te)} images ...")
    Str, ytr = extract_stats(tr), tr["label"].to_numpy(int)
    Ste, yte = extract_stats(te), te["label"].to_numpy(int)

    outdir = OUTPUT_DIR / "gmm_baseline"
    results = {}
    for name, cols in FEATURE_SETS.items():
        clf = GMMColorClassifier(args.components, args.seed).fit(Str[:, cols], ytr)
        m = image_level_metrics(yte, clf.predict(Ste[:, cols]))
        results[name] = m
        print(f"[gmm {name:>4} | {args.dataset} | k={args.components}] {summarize(m)}")
        save_json({"model": f"gmm_{name}", "dataset": args.dataset, "n_components": args.components,
                   "seed": args.seed, "features": STAT_NAMES[cols], "metrics": m},
                  outdir / f"gmm_{args.dataset}_k{args.components}_{name}.json")

    save_json({"dataset": args.dataset, "n_components": args.components,
               "compare": {n: {k: results[n][k] for k in
                               ("exact_accuracy", "within_1_stage_accuracy", "stage_mae",
                                "qwk", "macro_f1", "per_class_recall")} for n in results}},
              outdir / f"gmm_{args.dataset}_k{args.components}_compare.json")

    # Per-image color features as CSV — the extracted 'RGB points' (+ std/HSV/Lab), with metadata
    # and split, so the features can be re-plotted / re-analysed without re-running extraction.
    save_features_csv(tr, Str, te, Ste, outdir / f"gmm_{args.dataset}_features.csv")

    scatter_rgb(Str, ytr, outdir / f"gmm_{args.dataset}_rgb_scatter.png")
    scatter_lab(Str, ytr, outdir / f"gmm_{args.dataset}_lab_scatter.png")
    print(f"[gmm] saved metrics (rgb + rich + compare) + scatters -> {outdir}")


if __name__ == "__main__":
    main()
