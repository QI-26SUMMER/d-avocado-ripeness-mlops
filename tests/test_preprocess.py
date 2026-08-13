"""Tests for the real-photo preprocessing geometry core (src/inference/preprocess.py).

Regression coverage for the EXIF-orientation bug: phone photos are stored landscape with an
orientation tag, PIL does not apply it on open, but the segmenters do apply it internally before
predicting (rembg calls ImageOps.exif_transpose in remove(); see rembg/bg.py). The mask therefore
came back in display orientation while the image was still in stored orientation, and
apply_mask_white silently squashed the mismatched mask to fit — producing a confident-looking but
completely wrong crop rather than an error.

No segmenter is needed: the tests use a stub that mimics rembg by applying exif_transpose itself,
so the whole pipeline is exercised with only PIL + NumPy.

Run (either works):
  python -m pytest tests/test_preprocess.py -q
  python tests/test_preprocess.py
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.preprocess import (  # noqa: E402
    ASPECT_TOL,
    NoAvocadoDetected,
    align_mask_to_image,
    apply_mask_white,
    crop_pad_square,
    load_rgb,
    mask_to_bbox,
    preprocess_real_photo,
)

EXIF_ORIENTATION_TAG = 274
GREEN, RED = (0, 200, 0), (220, 0, 0)


# ─── helpers ────────────────────────────────────────────────────────────────
def _jpeg_with_orientation(img: Image.Image, orientation: int) -> Image.Image:
    """Round-trip `img` through JPEG bytes carrying an EXIF orientation tag, like a phone photo."""
    exif = img.getexif()
    exif[EXIF_ORIENTATION_TAG] = orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, exif=exif)
    buf.seek(0)
    return Image.open(buf)


def _half_and_half(size=(60, 30)) -> Image.Image:
    """Left half green (the 'avocado'), right half red (background that must not survive)."""
    w, h = size
    img = Image.new("RGB", size, RED)
    img.paste(Image.new("RGB", (w // 2, h), GREEN), (0, 0))
    return img


class _ExifApplyingSegmenter:
    """Stub standing in for rembg/InSPyReNet: applies EXIF orientation internally (as rembg's
    remove() does) and then segments by colour, so the mask is defined by image CONTENT. That
    makes a misaligned mask detectable — it would white out the wrong pixels."""

    def mask(self, image):
        arr = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        return (arr[:, :, 1] > 150) & (arr[:, :, 0] < 100)   # the green region


def _assert_raises(exc, fn, needle=""):
    try:
        fn()
    except exc as e:
        assert needle in str(e), f"expected {needle!r} in error message, got: {e}"
        return
    raise AssertionError(f"expected {exc.__name__} but nothing was raised")


def _count_reddish(img: Image.Image) -> int:
    arr = np.asarray(img.convert("RGB")).astype(int)
    return int(((arr[:, :, 0] > 150) & (arr[:, :, 1] < 100)).sum())


def _green_fraction(img: Image.Image) -> float:
    """How much of the frame the 'fruit' fills — the signal that the crop landed on it."""
    arr = np.asarray(img.convert("RGB")).astype(int)
    green = (arr[:, :, 1] > 150) & (arr[:, :, 0] < 100)
    return float(green.sum()) / green.size


# ─── load_rgb: the single orientation-normalisation point ───────────────────
def test_load_rgb_applies_exif_orientation():
    # orientation 6 = "rotate for display", so a 60x30 stored photo displays as 30x60.
    stored = _jpeg_with_orientation(_half_and_half((60, 30)), orientation=6)
    assert stored.size == (60, 30), "sanity: stored pixels stay landscape"
    assert load_rgb(stored).size == (30, 60), "load_rgb must return DISPLAY orientation"


def test_load_rgb_strips_the_orientation_tag():
    # Stripping is what makes the segmenter's own exif_transpose a harmless no-op.
    out = load_rgb(_jpeg_with_orientation(_half_and_half(), orientation=6))
    assert out.getexif().get(EXIF_ORIENTATION_TAG) is None
    assert ImageOps.exif_transpose(out).size == out.size, "second transpose must not rotate again"


def test_load_rgb_leaves_tagless_images_alone():
    plain = Image.new("RGB", (40, 20), GREEN)
    assert load_rgb(plain).size == (40, 20)
    assert load_rgb(np.zeros((20, 40, 3), np.uint8)).size == (40, 20)


def test_load_rgb_accepts_a_file_like_object():
    """serving/app.py hands it an io.BytesIO of the uploaded JPEG — keep that path working."""
    buf = io.BytesIO()
    tagged = _jpeg_with_orientation(_half_and_half((60, 30)), orientation=6)
    tagged.save(buf, format="JPEG", quality=95, exif=tagged.getexif())
    buf.seek(0)
    out = load_rgb(buf)
    assert out.size == (30, 60) and out.mode == "RGB"


# ─── align_mask_to_image: the guard that replaced the silent squash ─────────
def test_align_mask_passes_through_exact_match():
    img = Image.new("RGB", (40, 20))
    mask = np.ones((20, 40), bool)
    assert align_mask_to_image(mask, img) is mask


def test_align_mask_resamples_a_pure_rescale():
    # A segmenter predicting at a fixed lower resolution is legitimate: same aspect, so resample.
    img = Image.new("RGB", (4000, 2252))
    small = np.zeros((577, 1024), bool)
    small[288:, 512:] = True
    out = align_mask_to_image(small, img)
    assert out.shape == (2252, 4000)
    assert out[:, :100].sum() == 0 and out[-100:, -100:].all(), "resample must preserve placement"


def test_align_mask_raises_on_orientation_mismatch():
    """THE regression: a 4000x2252 image against a 2252x4000 mask must fail loudly."""
    img = Image.new("RGB", (4000, 2252))
    rotated_mask = np.ones((4000, 2252), bool)
    _assert_raises(ValueError, lambda: align_mask_to_image(rotated_mask, img), "orientation")


def test_align_mask_tolerance_is_not_absurdly_tight():
    # Off-by-one rounding in a rescale must not be mistaken for an orientation flip.
    img = Image.new("RGB", (4000, 2252))
    assert align_mask_to_image(np.ones((577, 1024), bool), img).shape == (2252, 4000)
    assert ASPECT_TOL < 0.1, "tolerance must stay tight enough to catch a real flip"


# ─── apply_mask_white: no longer absorbs a geometry bug ─────────────────────
def test_apply_mask_white_replaces_background_only():
    img = Image.new("RGB", (4, 2), RED)
    mask = np.zeros((2, 4), bool)
    mask[:, :2] = True
    arr = np.asarray(apply_mask_white(img, mask))
    assert (arr[:, :2] == RED).all(), "inside the mask is untouched"
    assert (arr[:, 2:] == 255).all(), "outside the mask becomes white"


def test_apply_mask_white_raises_on_shape_mismatch():
    _assert_raises(
        ValueError,
        lambda: apply_mask_white(Image.new("RGB", (40, 20)), np.ones((20, 39), bool)),
        "align_mask_to_image",
    )


# ─── geometry core ──────────────────────────────────────────────────────────
def test_mask_to_bbox_is_tight_then_margined():
    mask = np.zeros((100, 100), bool)
    mask[40:60, 30:50] = True
    assert mask_to_bbox(mask, margin_frac=0.0) == (30, 40, 50, 60)
    # 8% of a 20px box = 1.6 -> rounds to 2
    assert mask_to_bbox(mask, margin_frac=0.08) == (28, 38, 52, 62)


def test_mask_to_bbox_clamps_at_the_edges():
    mask = np.zeros((10, 10), bool)
    mask[0:10, 0:10] = True
    assert mask_to_bbox(mask, margin_frac=0.5) == (0, 0, 10, 10)


def test_mask_to_bbox_rejects_an_empty_mask():
    _assert_raises(NoAvocadoDetected, lambda: mask_to_bbox(np.zeros((5, 5), bool)))


def test_crop_pad_square_centres_without_distorting():
    img = Image.new("RGB", (100, 100), RED)
    img.paste(Image.new("RGB", (20, 10), GREEN), (10, 20))
    out = crop_pad_square(img, (10, 20, 30, 30))   # a 20x10 crop
    assert out.size == (20, 20), "padded to a square of the longer side"
    arr = np.asarray(out)
    assert (arr[0] == 255).all() and (arr[-1] == 255).all(), "padding is white"
    assert (arr[5:15] == GREEN).all(), "the crop sits centred, undistorted"


# ─── end to end ─────────────────────────────────────────────────────────────
def test_rotated_photo_produces_a_clean_crop():
    """Full pipeline on a photo whose EXIF says 'rotate me', against a segmenter that applies
    that rotation internally.

    The fill ratio is the assertion that matters. A misaligned mask still keeps the red out (it
    whites out *something*), but it crops the wrong region: measured against the old code this
    fell from 0.86 to 0.43 of the frame, i.e. half the fruit missing, with nothing raised.
    """
    photo = _jpeg_with_orientation(_half_and_half((60, 30)), orientation=6)
    out = preprocess_real_photo(photo, _ExifApplyingSegmenter(), img_size=64)

    assert out.size == (64, 64)
    assert _count_reddish(out) == 0, "background leaked into the crop — mask/image misaligned"
    assert _green_fraction(out) > 0.7, "the crop is not centred on the fruit — mask misaligned"


def test_tagged_photo_matches_manually_rotated_photo():
    """The invariant the fix buys: a photo carrying an orientation tag must preprocess to the
    same thing as the same photo with the rotation already baked into its pixels (which is what
    rotating the file by hand produced, back when that was the only workaround)."""
    base = _half_and_half((60, 30))
    tagged = _jpeg_with_orientation(base, orientation=6)
    baked = _jpeg_with_orientation(base.transpose(Image.ROTATE_270), orientation=1)

    seg = _ExifApplyingSegmenter()
    a = np.asarray(preprocess_real_photo(tagged, seg, img_size=64)).astype(int)
    b = np.asarray(preprocess_real_photo(baked, seg, img_size=64)).astype(int)
    # Not bit-identical: the two are JPEG-encoded in different pixel layouts. Measured max
    # difference is 2/255, so anything beyond a handful of levels is a real divergence.
    assert np.abs(a - b).max() <= 4, "tagged and pre-rotated photos preprocess differently"


def test_upright_photo_still_works():
    """orientation=1 needs no rotation — the path that already worked must not regress."""
    photo = _jpeg_with_orientation(_half_and_half((60, 30)), orientation=1)
    out = preprocess_real_photo(photo, _ExifApplyingSegmenter(), img_size=64)
    assert out.size == (64, 64)
    assert _count_reddish(out) == 0
    assert _green_fraction(out) > 0.7


def test_no_mask_raises_no_avocado_detected():
    class _Empty:
        def mask(self, image):
            return None

    _assert_raises(
        NoAvocadoDetected,
        lambda: preprocess_real_photo(Image.new("RGB", (40, 20)), _Empty()),
    )


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
