"""Image Dataset + split/loader construction (paper reproduction §3, §5).

The model input is a single RGB image only (§1). Metadata such as Storage Group/Day/Sample/
view/Time Stamp/days_left/color statistics must never go into the model input -- they're
only for filtering, splitting, aggregation, shelf-life, and analysis purposes.

The split reuses splits.csv produced by src/data/split.py (shared by all models).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from ..common.utils import OUTPUT_DIR, seed_worker
    from .data import IMAGE_DIR
except ImportError:  # src/ on sys.path (tests, serving): `..` would escape the top-level package
    from common.utils import OUTPUT_DIR, seed_worker
    from data.data import IMAGE_DIR

METADATA_CLEAN = OUTPUT_DIR / "splits" / "metadata_clean.csv"
SPLITS_CSV = OUTPUT_DIR / "splits" / "splits.csv"

# config's dataset name -> Storage Group filter (general=all)
DATASET_GROUPS = {
    "general": None,
    "T10": ["T10"],
    "T20": ["T20"],
    "Tam": ["Tam"],
    "Tamb": ["Tam"],   # paper's name Tamb == dataset's name Tam
}


def load_metadata_and_splits() -> pd.DataFrame:
    """Attach the Split column from splits.csv onto metadata_clean.csv and return it."""
    if not METADATA_CLEAN.exists() or not SPLITS_CSV.exists():
        raise SystemExit("Run `python -m src.data.validate_data` and `python -m src.data.split` first.")
    meta = pd.read_csv(METADATA_CLEAN)
    splits = pd.read_csv(SPLITS_CSV)
    merged = meta.merge(splits[["Storage Group", "Sample", "Split"]],
                        on=["Storage Group", "Sample"], how="left")
    assert merged["Split"].notna().all(), "Some images have no split assigned"
    return merged


def subset_frame(merged: pd.DataFrame, dataset: str, split: str) -> pd.DataFrame:
    """Subset by dataset(general/T10/T20/Tam) x split(train/val/test)."""
    if dataset not in DATASET_GROUPS:
        raise ValueError(f"unknown dataset {dataset!r}")
    groups = DATASET_GROUPS[dataset]
    df = merged if groups is None else merged[merged["Storage Group"].isin(groups)]
    return df[df["Split"] == split].reset_index(drop=True)


class AvocadoImageDataset(Dataset):
    """Returns only (image tensor, target 0-4). No metadata goes in (§1)."""

    def __init__(self, df: pd.DataFrame, image_dir: Path = IMAGE_DIR, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(self.image_dir / f"{row['File Name']}.jpg").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"]) - 1  # 1-5 -> 0-4


def make_loader(df, transform, batch_size, shuffle, num_workers=4,
                sampler_indices=None, generator=None) -> DataLoader:
    """Build a single DataLoader.

    sampler_indices: paper-style oversampled index list (train only). If given, iterates in that order.
    """
    ds = AvocadoImageDataset(df, transform=transform)
    common = dict(num_workers=num_workers,
                  worker_init_fn=seed_worker if num_workers > 0 else None,
                  persistent_workers=num_workers > 0)
    if sampler_indices is not None:
        from torch.utils.data import SubsetRandomSampler
        # Sampler that walks the (possibly duplicated) index list as-is, preserving duplicate indices
        from torch.utils.data import Sampler

        class _ListSampler(Sampler):
            def __init__(self, idx): self.idx = list(idx)
            def __iter__(self): return iter(self.idx)
            def __len__(self): return len(self.idx)

        return DataLoader(ds, batch_size=batch_size, sampler=_ListSampler(sampler_indices),
                          drop_last=True, generator=generator, **common)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle, generator=generator, **common)
