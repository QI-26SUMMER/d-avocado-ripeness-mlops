"""Generate and validate a Sample-level 70/15/15 split (paper reproduction §4).

Run: python -m src.split --seed 42
- Splits each Storage Group 70/15/15 at the Sample level (image-level random split forbidden, §2.1).
- General split = union of the per-storage-group splits (lets General/storage-group models be
  compared on the same test fruit -- an implementation choice).
- Output: outputs/paper_reproduction/splits/splits.csv (columns: Sample, Storage Group, Split, Random Seed)
- Generated once, then reused by every model. Includes validation asserts.

The paper only states that 70/15/15 was assigned randomly via a Python script (no leakage at the
Sample level). The split seed is not disclosed in the paper -> implementation choice (specify via
config/CLI).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .utils import OUTPUT_DIR
except ImportError:
    from utils import OUTPUT_DIR

SPLITS_CSV = OUTPUT_DIR / "splits" / "splits.csv"
METADATA_CLEAN = OUTPUT_DIR / "splits" / "metadata_clean.csv"


def make_splits(meta: pd.DataFrame, seed: int = 42,
                frac=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    """Sample-level 70/15/15 per storage group. Label distribution is approximated by random assignment (§4)."""
    assert abs(sum(frac) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    rows = []
    # Unique individuals at the (Storage Group, Sample) level
    samples = meta[["Storage Group", "Sample"]].drop_duplicates()
    for grp, sub in samples.groupby("Storage Group"):
        ids = sub["Sample"].to_numpy().copy()  # pandas 3.0: to_numpy() is read-only
        rng.shuffle(ids)
        n = len(ids)
        n_tr = round(frac[0] * n)
        n_va = round(frac[1] * n)
        # test = the remainder (so the total sums exactly to n)
        assign = (["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va))
        for s, sp in zip(ids, assign):
            rows.append({"Sample": int(s), "Storage Group": grp,
                         "Split": sp, "Random Seed": seed})
    return pd.DataFrame(rows).sort_values(["Storage Group", "Sample"]).reset_index(drop=True)


def validate_splits(splits: pd.DataFrame, meta: pd.DataFrame) -> None:
    """§4/§17 validation: zero individual overlap, each Sample in exactly one split, General=union of groups."""
    key = splits["Storage Group"].astype(str) + "_" + splits["Sample"].astype(str)
    by_split = {sp: set(key[splits["Split"] == sp]) for sp in ["train", "val", "test"]}
    # 1) Zero overlap
    assert by_split["train"] & by_split["val"] == set()
    assert by_split["train"] & by_split["test"] == set()
    assert by_split["val"] & by_split["test"] == set()
    # 2) Each (group, sample) belongs to exactly one split
    assert (splits.groupby(["Storage Group", "Sample"])["Split"].nunique() == 1).all()
    assert len(splits) == meta[["Storage Group", "Sample"]].drop_duplicates().shape[0]
    # 3) General split = union of per-storage-group splits (trivially true by construction, verified explicitly)
    #    train individuals in General == union of train individuals across storage groups
    general_train = set(key[splits["Split"] == "train"])
    union_train = set()
    for grp in splits["Storage Group"].unique():
        m = (splits["Storage Group"] == grp) & (splits["Split"] == "train")
        union_train |= set(key[m])
    assert general_train == union_train
    # 4) At the image level too, every image of a sample falls in exactly one split
    merged = meta.merge(splits[["Storage Group", "Sample", "Split"]],
                        on=["Storage Group", "Sample"], how="left")
    assert merged["Split"].notna().all(), "Some images have no split assigned"
    assert (merged.groupby(["Storage Group", "Sample"])["Split"].nunique() == 1).all()


def summary(splits: pd.DataFrame, meta: pd.DataFrame) -> str:
    merged = meta.merge(splits[["Storage Group", "Sample", "Split"]],
                        on=["Storage Group", "Sample"], how="left")
    lines = ["=== split summary (individual count / image count) ==="]
    # Individual counts by storage group x split
    samp = splits.groupby(["Storage Group", "Split"]).size().unstack(fill_value=0)[
        ["train", "val", "test"]]
    lines.append("[individuals]\n" + samp.to_string())
    img = merged.groupby(["Split"]).size().reindex(["train", "val", "test"])
    lines.append(f"[images] train {img['train']} / val {img['val']} / test {img['test']}")
    # General label distribution (image-based)
    dist = (merged.groupby("Split")["label"].value_counts(normalize=True)
            .unstack().reindex(["train", "val", "test"])[[1, 2, 3, 4, 5]].round(3))
    lines.append("[General label 1-5 ratio]\n" + dist.to_string())
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not METADATA_CLEAN.exists():
        raise SystemExit(f"Run `python -m src.validate_data` first: {METADATA_CLEAN} not found")
    meta = pd.read_csv(METADATA_CLEAN)

    splits = make_splits(meta, seed=args.seed)
    validate_splits(splits, meta)
    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(SPLITS_CSV, index=False, encoding="utf-8")

    print(summary(splits, meta))
    print(f"\nValidation passed (zero individual overlap, General=union of storage groups). splits.csv -> {SPLITS_CSV}")


if __name__ == "__main__":
    main()
