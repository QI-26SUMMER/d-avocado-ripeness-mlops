"""Data loading / splitting / label derivation.

Enforces the pitfalls from CLAUDE.md §2 in code:
  - §2.1 individual-level split (image-level split X)
  - §2.2 verify Sample global uniqueness -> group key
  - §2.6 derive days_left + right-censoring flag
  - §3 skeleton: filter out 12 missing images

Facts verified by EDA (no guessing, CLAUDE.md §1):
  images 14,710 / metadata rows 14,722 / missing 12 / individuals 478 / labels 1-5 ordinal.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

# pandas is used at runtime only inside load_metadata (imported lazily there). Everywhere else it
# appears only in type hints (strings under `from __future__ import annotations`), so this module
# stays importable without pandas.
if TYPE_CHECKING:
    import pandas as pd

# --- Paths -------------------------------------------------------------------
# These are LOCAL paths (mirrors utils.OUTPUT_DIR's env-override style). On GCP the training
# entrypoint downloads the dataset from GCS to local SSD first and points the env vars below at
# it, so by the time this module reads anything the paths are always local. Defined without
# importing utils on purpose, to keep this module dependency-free.
# parents[2] because this file is src/data/data.py — two levels below the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AVOCADO_DATA_DIR", str(REPO_ROOT / "data")))
META_XLSX = DATA_DIR / "Avocado Ripening Dataset.xlsx"

# xlsx was moved to the data/ root. Images live under one nested layer of folders, per CLAUDE.md §1.
# Default is the one-level-nested local path from CLAUDE.md §1. When running on GCP (Vertex),
# the container copies the images to local SSD and points AVOCADO_IMAGE_DIR at that path
# (no effect on local runs).
IMAGE_DIR = Path(
    os.environ.get(
        "AVOCADO_IMAGE_DIR",
        str(
            DATA_DIR
            / "Hass Avocado Ripening Photographic Dataset"
            / "Hass Avocado Ripening Photographic Dataset"
            / "Avocado Ripening Dataset"
        ),
    )
)

# Curated keep-list manifest (avocado_complete_states.csv): only samples with a complete
# 1->5 trajectory. On GCP the entrypoint points AVOCADO_MANIFEST at the /gcs-mounted copy;
# locally it defaults to the repo's data/ copy. Absent -> no exclusion (full dataset).
MANIFEST_CSV = Path(os.environ.get("AVOCADO_MANIFEST", str(DATA_DIR / "avocado_complete_states.csv")))

# Label definitions moved to common/labels.py (LABELS, NUM_CLASSES): they are domain constants
# needed by models/ and serving too, and keeping them here made common/ depend on this package.

FILENAME_RE = re.compile(r"^(T\w+)_d(\d+)_(\d+)_([ab])_(\d+)$")


def load_metadata(xlsx: Path = META_XLSX) -> pd.DataFrame:
    """Read the DATABASE sheet and attach columns parsed from the file name."""
    import pandas as pd  # lazy: only training/EDA needs pandas, not the serving container

    df = pd.read_excel(xlsx)  # only one sheet (DATABASE)
    df["Time Stamp"] = pd.to_datetime(df["Time Stamp"])

    parsed = df["File Name"].astype(str).str.extract(FILENAME_RE)
    if parsed.isna().any().any():
        raise ValueError("Some rows don't match the file name convention. Check the file name rule in CLAUDE.md §1.")
    df["group"] = parsed[0]
    df["day"] = parsed[1].astype(int)
    df["sample"] = parsed[2].astype(int)
    df["view"] = parsed[3]
    df["label"] = df["Ripening Index Classification"].astype(int)

    # File name <-> column consistency (verified: 100% match across 14,722 rows)
    assert (df["group"] == df["Storage Group"]).all()
    assert (df["day"] == df["Day of Experiment"]).all()
    assert (df["sample"] == df["Sample"]).all()
    assert (parsed[4].astype(int) == df["label"]).all()
    return df


def filter_existing_images(df: pd.DataFrame, image_dir: Path = IMAGE_DIR) -> pd.DataFrame:
    """§3: explicitly filter out the 12 missing images instead of letting them fail silently."""
    present = {p.stem for p in image_dir.glob("*.jpg")}
    keep = df["File Name"].astype(str).isin(present)
    dropped = int((~keep).sum())
    if dropped:
        print(f"[data] Excluded {dropped} missing image(s) (present in metadata but no file on disk)")
    return df[keep].reset_index(drop=True)


def filter_to_manifest(df: pd.DataFrame, manifest: Path = MANIFEST_CSV) -> pd.DataFrame:
    """Keep only rows whose File Name is in the curated keep-list manifest.

    The manifest (avocado_complete_states.csv) lists exactly the images to train on:
    only samples with a complete 1->5 trajectory. Samples that never reached stage 5
    or have a gap in the middle were removed by hand upstream.

    ⚠ CLAUDE.md §2.6: dropping never-reached-5 (right-censored) individuals biases the
    training set toward fast-ripeners. This exclusion is applied ONLY because a curated
    manifest is present — unset AVOCADO_MANIFEST (or remove the file) to train on the
    full dataset. No-op when the manifest is absent, so full-dataset runs are unchanged.
    """
    import pandas as pd  # lazy: keep the serving container's pandas-free import contract

    if not Path(manifest).exists():
        print(f"[data] No manifest at {manifest} -> training on full dataset (no exclusion)")
        return df
    keep = set(pd.read_csv(manifest)["File Name"].astype(str))
    mask = df["File Name"].astype(str).isin(keep)
    kept = df[mask].reset_index(drop=True)
    n_s0 = df.groupby(["Storage Group", "Sample"]).ngroups
    n_s1 = kept.groupby(["Storage Group", "Sample"]).ngroups
    print(f"[data] Manifest {Path(manifest).name}: kept {len(kept)}/{len(df)} images, "
          f"{n_s1}/{n_s0} samples (excluded {n_s0 - n_s1} samples: never-reached-5 / gap)")
    return kept


def group_key(df: pd.DataFrame) -> pd.Series:
    """§2.2: verify Sample is globally unique, then build the split key.

    Verification shows Sample is globally unique (0 reuse), but we still use the
    (group, sample) tuple defensively.
    """
    assert df.groupby("Sample")["Storage Group"].nunique().max() == 1, (
        "Sample ID reused across groups -> Storage Group must be part of the group key (CLAUDE.md §2.2)"
    )
    return df["Storage Group"].astype(str) + "_" + df["Sample"].astype(str)


def split_by_individual(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """§2.1: individual-level split. Must verify no leakage after splitting."""
    from sklearn.model_selection import GroupShuffleSplit

    groups = group_key(df)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    # Leakage check (CLAUDE.md §2.1)
    assert set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]) == set(), "Individual leakage!"
    return train.reset_index(drop=True), test.reset_index(drop=True)


def derive_days_left(df: pd.DataFrame) -> pd.DataFrame:
    """§2.6: days_left isn't in the dataset -- derive it. Leaves a right-censoring flag.

    End of shelf life = the first Day of Experiment on which label 5 is observed for an individual.
    Individuals that never reach label 5 (verified: 68/478) get censored=True, days_left=NaN.
    Simply dropping them would introduce a "slow-ripening individual" selection bias (§2.6).
    """
    end = (
        df[df["label"] == 5]
        .groupby(["Storage Group", "Sample"])["Day of Experiment"]
        .min()
        .rename("end_day")
    )
    out = df.merge(end, on=["Storage Group", "Sample"], how="left")
    out["censored"] = out["end_day"].isna()
    out["days_left"] = out["end_day"] - out["Day of Experiment"]  # NaN for censored individuals
    return out


if __name__ == "__main__":
    df = load_metadata()
    df = filter_existing_images(df)
    print(f"images {len(df)} / individuals {group_key(df).nunique()}")
    tr, te = split_by_individual(df)
    print(f"train {len(tr)} / test {len(te)} (individual-level, leakage check passed)")
    dl = derive_days_left(df)
    print(f"Right-censored individual-sessions: {int(dl['censored'].sum())}")
