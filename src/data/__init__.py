"""Dataset preparation: read the metadata, check it, and split it by individual.

Run in this order (the training entrypoint does exactly this):

    python -m src.data.validate_data     # integrity checks -> metadata_clean.csv
    python -m src.data.split             # 70/15/15 sample-level split -> splits.csv

    data.py           metadata loader, missing-image filter, curated manifest, group split
    validate_data.py  12 integrity checks -> metadata_clean.csv
    split.py          sample-level 70/15/15 -> splits.csv (generated once, reused by every model)
    dataset.py        torch Dataset/DataLoader (single image in, no metadata — CLAUDE.md §2.3)

CLAUDE.md §2.1 lives here: splits are built on the (Storage Group, Sample) key, never
image-level, and every split is asserted leak-free before it is written.
"""
