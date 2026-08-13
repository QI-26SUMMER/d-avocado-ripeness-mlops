"""Pipeline validation tests (paper reproduction §17, 11 assertions).

Run (either works):
  python -m pytest tests/test_pipeline.py -q
  python tests/test_pipeline.py            # also runs without pytest

Prerequisite: python -m src.data.validate_data ; python -m src.data.split --seed 42
Some tests (model forward / checkpoint) actually use images and torch.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import AvocadoImageDataset, load_metadata_and_splits, subset_frame  # noqa: E402
from common.shelf_life import ALPHA_5STAGE, estimate_days_left  # noqa: E402
from common.transforms import evaluation_transform, paper_train_transform  # noqa: E402
from training.sampler import effective_class_counts, paper_oversample_indices  # noqa: E402
from common.utils import OUTPUT_DIR  # noqa: E402

SPLITS = pd.read_csv(OUTPUT_DIR / "splits" / "splits.csv")
META = pd.read_csv(OUTPUT_DIR / "splits" / "metadata_clean.csv")
MERGED = load_metadata_and_splits()


def test_01_no_sample_in_multiple_splits():
    key = SPLITS["Storage Group"].astype(str) + "_" + SPLITS["Sample"].astype(str)
    tr = set(key[SPLITS.Split == "train"]); va = set(key[SPLITS.Split == "val"]); te = set(key[SPLITS.Split == "test"])
    assert tr & va == set() and tr & te == set() and va & te == set()
    assert (SPLITS.groupby(["Storage Group", "Sample"]).Split.nunique() == 1).all()


def test_02_general_equals_union_of_groups():
    for sp in ["train", "val", "test"]:
        key = SPLITS["Storage Group"].astype(str) + "_" + SPLITS["Sample"].astype(str)
        general = set(key[SPLITS.Split == sp])
        union = set()
        for grp in SPLITS["Storage Group"].unique():
            m = (SPLITS["Storage Group"] == grp) & (SPLITS.Split == sp)
            union |= set(key[m])
        assert general == union


def test_03_eval_transform_is_deterministic_train_is_not():
    from PIL import Image
    img = Image.new("RGB", (300, 300), (120, 150, 90))
    ev = evaluation_transform(224)
    a, b = ev(img), ev(img)
    assert np.allclose(a.numpy(), b.numpy()), "eval transform must be deterministic (no augmentation)"
    tr = paper_train_transform(224)
    import torch
    torch.manual_seed(0); x1 = tr(img)
    torch.manual_seed(1); x2 = tr(img)
    assert not np.allclose(x1.numpy(), x2.numpy()), "paper train transform must apply random augmentation"


def test_04_oversampling_balances_train_only():
    train_df = subset_frame(MERGED, "general", "train")
    idx = paper_oversample_indices(train_df["label"].values, seed=42)
    eff = effective_class_counts(train_df["label"].values, idx)
    assert len(set(eff.values())) == 1, f"classes not balanced after oversampling: {eff}"
    # val/test keep the original distribution (no duplication) - this function is used on train only (contract)
    assert len(idx) >= len(train_df)


def test_05_06_model_input_is_image_only_no_metadata():
    train_df = subset_frame(MERGED, "general", "train")
    ds = AvocadoImageDataset(train_df.iloc[:3], transform=evaluation_transform(224))
    item = ds[0]
    assert len(item) == 2, "Dataset must return only (image, label)"
    img, label = item
    assert tuple(img.shape) == (3, 224, 224), "input is a 3-channel RGB image only (no metadata channels concatenated)"
    assert isinstance(label, int)


def test_07_label_1to5_maps_to_0to4():
    train_df = subset_frame(MERGED, "general", "train")
    for L in [1, 2, 3, 4, 5]:
        sub = train_df[train_df.label == L].head(1)
        if len(sub) == 0:
            continue
        ds = AvocadoImageDataset(sub, transform=evaluation_transform(224))
        _, target = ds[0]
        assert target == L - 1


def test_08_two_side_grouping_by_sample_and_day():
    test_df = subset_frame(MERGED, "general", "test")
    g = test_df.groupby(["Storage Group", "Sample", "Day of Experiment"])["view"]
    assert g.apply(lambda s: set(s) <= {"a", "b"}).all()
    assert (g.size() <= 2).all(), "at most 2 sides (a/b) per observation"
    n_obs = test_df.drop_duplicates(["Storage Group", "Sample", "Day of Experiment"]).shape[0]
    assert n_obs == len(g)


def test_09_shelf_life_alpha_per_group():
    assert ALPHA_5STAGE["T10"] == -4.390
    assert ALPHA_5STAGE["T20"] == -2.116
    assert ALPHA_5STAGE["Tam"] == -1.929 == ALPHA_5STAGE["Tamb"]
    assert estimate_days_left(5, "T20") == 0.0
    assert abs(estimate_days_left(3, "T20") - 4.232) < 1e-9


def test_10_missing_images_not_in_dataset():
    import json
    rep = json.loads((OUTPUT_DIR / "reports" / "data_validation.json").read_text(encoding="utf-8"))
    missing = set(rep["3_missing_images"])
    assert len(missing) == 12
    assert missing & set(META["File Name"]) == set(), "a missing image is present in metadata_clean"


def test_11_checkpoint_reload_same_prediction():
    import torch
    from common.models import build_model
    ckpts = list((OUTPUT_DIR / "checkpoints").glob("*/best.pt"))
    if not ckpts:
        print("  [test_11 skip] no checkpoint found")
        return
    ck_path = ckpts[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = torch.load(ck_path, map_location=device, weights_only=False)["cfg"]
    test_df = subset_frame(MERGED, cfg["dataset"], "test").iloc[:16]
    ds = AvocadoImageDataset(test_df, transform=evaluation_transform(cfg.get("img_size", 224)))
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)

    def load_and_pred():
        m = build_model(cfg["model"], pretrained=False).to(device)
        m.load_state_dict(torch.load(ck_path, map_location=device, weights_only=False)["model"])
        m.eval()
        with torch.no_grad():
            return m(x).argmax(1).cpu().numpy()

    assert np.array_equal(load_and_pred(), load_and_pred()), "predictions differ after reload"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
