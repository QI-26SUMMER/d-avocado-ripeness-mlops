"""Data integrity checks (paper reproduction §2) + metadata_clean.csv generation.

Run: python -m src.data.validate_data
Runs 12 checks and saves the results to outputs/paper_reproduction/reports/data_validation.json,
then generates outputs/paper_reproduction/splits/metadata_clean.csv using only rows whose image
file actually exists on disk.

Expected values (trust the actual check over this if they differ, and report the discrepancy):
Sample 478 / images 14,710 / metadata-file gap ~12.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from ..common.utils import OUTPUT_DIR, save_json
    from .data import IMAGE_DIR, filter_to_manifest, load_metadata
except ImportError:  # src/ on sys.path (tests, serving): `..` would escape the top-level package
    from common.utils import OUTPUT_DIR, save_json
    from data.data import IMAGE_DIR, filter_to_manifest, load_metadata


def run_checks(image_check: bool = True) -> tuple[pd.DataFrame, dict]:
    df = load_metadata()  # includes file name parsing + consistency asserts
    present = {p.stem for p in Path(IMAGE_DIR).glob("*.jpg")}
    df_has = df["File Name"].astype(str).isin(present)

    report: dict = {}
    # 1. Number of metadata rows
    report["1_metadata_rows"] = int(len(df))
    # 2. Number of actual image files
    report["2_image_files"] = int(len(present))
    # 3. List of images that don't exist
    missing = df.loc[~df_has, "File Name"].astype(str).tolist()
    report["3_missing_images_count"] = len(missing)
    report["3_missing_images"] = missing
    # 4. Duplicate file names
    dup = df["File Name"].value_counts()
    report["4_duplicate_filenames"] = dup[dup > 1].to_dict()
    # 5. Number of unique Sample IDs
    report["5_unique_samples"] = int(df["Sample"].nunique())
    # 6. Number of samples per Storage Group
    report["6_samples_per_group"] = (
        df.groupby("Storage Group")["Sample"].nunique().to_dict())
    # 7. Label range 1-5
    report["7_label_range"] = [int(df["label"].min()), int(df["label"].max())]
    report["7_label_in_1_5"] = bool(df["label"].between(1, 5).all())
    # 8. Whether a Sample belongs to more than one Storage Group
    multi = df.groupby("Sample")["Storage Group"].nunique()
    report["8_samples_in_multiple_groups"] = int((multi > 1).sum())
    # 9. Whether view a/b parsed correctly
    report["9_view_values"] = df["view"].value_counts().to_dict()
    report["9_view_ok"] = bool(set(df["view"].unique()) <= {"a", "b"})
    # 10. Number of individuals with a non-monotonic (regressing) label sequence over time
    nonmono = 0
    for _, sub in df.groupby(["Storage Group", "Sample"]):
        s = sub.sort_values("Day of Experiment")["label"].values
        if (s[1:] < s[:-1]).any():
            nonmono += 1
    report["10_nonmonotonic_samples"] = nonmono
    # 11. Whether both a/b views exist for the same Sample on the same day
    vg = df.groupby(["Storage Group", "Sample", "Day of Experiment"])["view"].apply(
        lambda s: tuple(sorted(set(s))))
    report["11_view_pairs"] = {str(k): int(v) for k, v in vg.value_counts().items()}
    report["11_observations_missing_a_or_b"] = int((vg != ("a", "b")).sum())
    # 12. Corrupt images (fail to open)
    corrupt = []
    if image_check:
        from PIL import Image
        for fn in df.loc[df_has, "File Name"].astype(str):
            try:
                Image.open(Path(IMAGE_DIR) / f"{fn}.jpg").verify()
            except Exception:
                corrupt.append(fn)
    report["12_image_check_run"] = image_check
    report["12_corrupt_images_count"] = len(corrupt)
    report["12_corrupt_images"] = corrupt

    # metadata_clean = file actually exists + not corrupt ...
    clean_mask = df_has & (~df["File Name"].astype(str).isin(set(corrupt)))
    clean = df.loc[clean_mask].reset_index(drop=True)
    report["metadata_clean_rows_before_manifest"] = int(len(clean))
    # ... then restrict to the curated keep-list manifest (excludes never-reached-5 /
    # gap samples; no-op if the manifest is absent). See data.filter_to_manifest + §2.6.
    clean = filter_to_manifest(clean)
    report["metadata_clean_rows"] = int(len(clean))
    report["metadata_clean_samples"] = int(clean.groupby(["Storage Group", "Sample"]).ngroups)
    return clean, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-image-check", action="store_true",
                    help="Skip the full corrupt-image scan (check 12) for speed")
    args = ap.parse_args()

    clean, report = run_checks(image_check=not args.skip_image_check)

    (OUTPUT_DIR / "splits").mkdir(parents=True, exist_ok=True)
    clean_cols = ["File Name", "Time Stamp", "Storage Group", "Sample",
                  "Day of Experiment", "Ripening Index Classification",
                  "group", "day", "view", "label"]
    clean_path = OUTPUT_DIR / "splits" / "metadata_clean.csv"
    clean[clean_cols].to_csv(clean_path, index=False, encoding="utf-8")
    save_json(report, OUTPUT_DIR / "reports" / "data_validation.json")

    print("=== Data Integrity Checks ===")
    for k in ["1_metadata_rows", "2_image_files", "3_missing_images_count",
              "5_unique_samples", "6_samples_per_group", "7_label_range",
              "8_samples_in_multiple_groups", "9_view_ok",
              "10_nonmonotonic_samples", "11_observations_missing_a_or_b",
              "12_corrupt_images_count", "metadata_clean_rows_before_manifest",
              "metadata_clean_rows", "metadata_clean_samples"]:
        print(f"  {k}: {report[k]}")
    if report["4_duplicate_filenames"]:
        print(f"  4_duplicate_filenames: {report['4_duplicate_filenames']}")
    else:
        print("  4_duplicate_filenames: none")
    print(f"\nmetadata_clean.csv -> {clean_path} ({len(clean)} rows)")
    print(f"Report -> {OUTPUT_DIR / 'reports' / 'data_validation.json'}")


if __name__ == "__main__":
    main()
