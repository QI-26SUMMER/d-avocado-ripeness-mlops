"""Real-photo preprocessing pipeline (INFERENCE ONLY, on kitchen photos): segment the avocado →
erase the background to white using the mask → crop → pad to a white square → resize to the
classifier input. Training images are left untouched (they are already light-box).

Why. CLAUDE.md §3 domain gap: the classifier is trained on light-box photos (avocado centered
on a matte-white background), but users shoot with a phone under kitchen lighting with clutter
around the fruit. We replace every non-avocado pixel with white (mask-based background removal)
and re-center the fruit on a white square, reproducing the light-box framing so the classifier
sees an in-distribution image. §3 forbids background-uniformity tricks (threshold segmentation)
— a learned segmenter does not rely on a clean background, so it is allowed.

Segmenter is SWAPPABLE (this is the whole design). Anything with `.mask(pil) -> bool ndarray|None`
plugs into the same geometry core. Three backends:
  - InSPyReNetSegmenter (transparent-background, mode='base') ← CURRENT DEFAULT (2026-07). Best
    general edge quality of the CPU-viable options tested (u2net / isnet-general-use / birefnet-
    general / InSPyReNet all compared on real scans) — chosen for overall mask quality, NOT
    because it fixes any one specific case. ~370 MB checkpoint, ~20-22 s/image on CPU (mode=
    'base', 1024px); 'fast' mode (384px) is ~3 s/image but was not what was chosen here. NOTE:
    on a real scan set, u2net / isnet-general-use / birefnet-general / InSPyReNet ALL agreed on
    including a linear skin discoloration on one test fruit — four independent architectures
    agreeing means that mark is real fruit surface, not a segmentation error; do not expect a
    future segmenter swap to remove genuine blemishes/marks that are actually on the avocado.
  - RembgSegmenter (u2net)  ← fast/light alternative. ~176 MB, ~1.6 s/image, no gated weights.
    Segments the most prominent foreground object — fine when the avocado is the subject, but
    not avocado-specific. Was the default before InSPyReNet; kept as the low-latency fallback.
  - Sam3Segmenter (SAM3)    ← FUTURE SWAP. Concept prompt "avocado" (precise, avocado-only) but
    heavy (~3.5 GB), GPU-only-practical, and its weights are gated on Hugging Face.

Orientation. Phone photos are usually stored landscape with an EXIF orientation tag telling the
viewer to rotate (6/8 = portrait). PIL does NOT apply that tag on open, but the segmenters DO
apply it internally before predicting (rembg calls ImageOps.exif_transpose in remove(); see
rembg/bg.py). So the mask came back in display orientation while the image was still in stored
orientation, and the two disagreed about the photo's geometry — a 4000x2252 image against a
2252x4000 mask. load_rgb() now normalises orientation up front for everyone; exif_transpose is
idempotent (it strips the tag it consumed), so the segmenter's own call becomes a no-op.

The geometry core (mask → bbox → crop → pad → resize) is pure PIL/NumPy and unit-testable
without any segmenter. No avocado found → NoAvocadoDetected (serving turns this into the
backend's {"error": ...} → 422 NO_AVOCADO_DETECTED). This module does NOT decide where it runs.

Usage:
    from preprocess import load_segmenter, preprocess_real_photo
    seg = load_segmenter("inspyrenet")            # once (default); "rembg"/"sam3" to swap
    img224 = preprocess_real_photo(pil_img, seg, img_size=224)   # PIL.Image, 224x224
    # then the SAME transform the model trained with: evaluation_transform(224)(img224)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # keep torch/PIL/rembg/ultralytics out of import time for the pure core
    from PIL import Image

CONCEPT = "avocado"          # SAM3 promptable-concept noun phrase (unused by rembg)
DEFAULT_MARGIN_FRAC = 0.08   # expand the mask bbox by this fraction of its size before cropping
WHITE = (255, 255, 255)      # light-box background (matches paper_train_transform fill=255)
# Relative tolerance when comparing mask vs image aspect ratio (align_mask_to_image). Generous:
# a pure rescale keeps the ratio to well under 1% (4000x2252 -> 1024x577 is 0.09% off), while an
# orientation mismatch flips it entirely (1.776 vs 0.563).
ASPECT_TOL = 0.02


class NoAvocadoDetected(Exception):
    """Raised when the segmenter finds no avocado (→ backend NO_AVOCADO_DETECTED)."""


# ─── Segmenters (swappable; each exposes .mask(pil) -> bool ndarray | None) ──
class InSPyReNetSegmenter:
    """CURRENT DEFAULT. InSPyReNet salient-object segmentation (transparent-background package,
    the paper's reference implementation). Best general edge quality of the options compared on
    real scans (see module docstring). mode='base' (1024px) is the higher-quality, ~20-22 s/image
    setting on CPU; mode='fast' (384px) trades quality for ~3 s/image. Downloads a ~370 MB
    checkpoint on first use (cached under ~/.transparent-background, or
    $TRANSPARENT_BACKGROUND_FILE_PATH/.transparent-background if that env is set)."""

    def __init__(self, mode: str = "base", device: str | None = None):
        from transparent_background import Remover

        self._remover = Remover(mode=mode, device=device)

    def mask(self, image) -> np.ndarray | None:
        m = self._remover.process(load_rgb(image), type="map", threshold=0.5)
        arr = np.asarray(m)[..., 0] > 127
        return arr if arr.any() else None


class RembgSegmenter:
    """Fast/light alternative (was the default before InSPyReNet). rembg salient-object
    segmentation (u2net). CPU-friendly, no gated weights. Segments the most prominent foreground
    object (good when the avocado is the subject). Downloads the u2net model on first use
    (~176 MB).

    Caveat: it does NOT verify the object is an avocado, so it cannot reliably reject
    non-avocado images — a blank/uniform frame returns a bogus ~50% centred blob, not None.
    So NO_AVOCADO_DETECTED is effectively disabled under rembg; meaningful rejection needs the
    concept-aware Sam3Segmenter. mask() returns None only when nothing foreground-like is found
    (e.g. pure noise)."""

    def __init__(self, model_name: str = "u2net"):
        from rembg import new_session

        self._session = new_session(model_name)

    def mask(self, image) -> np.ndarray | None:
        from rembg import remove

        m = remove(load_rgb(image), session=self._session, only_mask=True, post_process_mask=True)
        arr = np.asarray(m) > 127
        return arr if arr.any() else None


class Sam3Segmenter:
    """FUTURE SWAP. SAM3 promptable-concept segmentation, prompt "avocado" (precise, avocado-
    only). Heavy (~3.5 GB), GPU-only-practical, weights gated on HF (manual sam3.pt download)."""

    def __init__(self, weights: str = "sam3.pt", device: str | None = None, conf: float = 0.25):
        import torch
        from ultralytics.models.sam import SAM3SemanticPredictor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[preprocess] WARNING: SAM3 on CPU is very slow (~seconds+/image); use a GPU.")
        self._predictor = SAM3SemanticPredictor(overrides=dict(
            conf=conf, task="segment", mode="predict", model=weights,
            device=device, save=False, verbose=False))

    def mask(self, image, concept: str = CONCEPT) -> np.ndarray | None:
        self._predictor.set_image(_to_ndarray(image))
        masks = _extract_masks(self._predictor(text=[concept]))
        return max(masks, key=lambda m: int(m.sum())) if masks else None  # largest instance


def load_segmenter(backend: str = "inspyrenet", **kwargs):
    """Factory for the swappable segmenter.

    backend='inspyrenet' (default, CPU, best quality) | 'rembg' (CPU, fast/light) | 'sam3' (GPU,
    avocado-concept precise). SWAP the default here (and nowhere else) to change what every
    caller gets. kwargs pass through (inspyrenet: mode, device; rembg: model_name; sam3: weights,
    device, conf).
    """
    backend = backend.lower()
    if backend == "inspyrenet":
        return InSPyReNetSegmenter(**kwargs)
    if backend == "rembg":
        return RembgSegmenter(**kwargs)
    if backend == "sam3":
        return Sam3Segmenter(**kwargs)
    raise ValueError(f"unknown segmenter backend {backend!r} (expected 'inspyrenet'|'rembg'|'sam3')")


def _extract_masks(results) -> list[np.ndarray]:
    """Pull binary masks out of an Ultralytics results object (SAM3, defensive across shapes)."""
    res = results[0] if isinstance(results, (list, tuple)) else results
    m = getattr(res, "masks", None)
    if m is None or getattr(m, "data", None) is None:
        return []
    data = m.data
    arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
    return [arr[i].astype(bool) for i in range(arr.shape[0])]


# ─── Pure geometry core (no segmenter needed — unit-testable) ───────────────
def mask_to_bbox(mask: np.ndarray, margin_frac: float = DEFAULT_MARGIN_FRAC) -> tuple[int, int, int, int]:
    """Tight (left, top, right, bottom) around the mask's True pixels, expanded by
    margin_frac of the box size and clamped to the image. Raises if the mask is empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise NoAvocadoDetected("empty mask")
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    mx = int(round((x1 - x0) * margin_frac))
    my = int(round((y1 - y0) * margin_frac))
    return (max(x0 - mx, 0), max(y0 - my, 0), min(x1 + mx, w), min(y1 + my, h))


def align_mask_to_image(mask: np.ndarray, image) -> np.ndarray:
    """Put `mask` in the image's pixel coordinates, or raise if the two disagree on geometry.

    A segmenter may legitimately return the mask at a different RESOLUTION than the input (some
    models predict at a fixed size); that is a pure rescale and is resampled here. A different
    ASPECT RATIO is never legitimate — it means the mask and the image disagree about the photo's
    shape, in practice because EXIF orientation was applied to one and not the other (a 4000x2252
    photo against a 2252x4000 mask). That case used to be silently squashed to fit, which produced
    a confident-looking but completely wrong crop instead of an error, so it raises now.

    Call this ONCE after segmenting: mask_to_bbox and apply_mask_white then share one coordinate
    space, so the bbox indexes the same pixels that got whited out.
    """
    from PIL import Image

    iw, ih = image.size
    mh, mw = mask.shape
    if (mh, mw) == (ih, iw):
        return mask
    img_aspect, mask_aspect = iw / ih, mw / mh
    if abs(mask_aspect - img_aspect) > ASPECT_TOL * img_aspect:
        raise ValueError(
            f"mask/image geometry mismatch: mask is {mw}x{mh} (aspect {mask_aspect:.3f}) but the "
            f"image is {iw}x{ih} (aspect {img_aspect:.3f}), differing by more than "
            f"{ASPECT_TOL:.0%}. They disagree about the photo's orientation — most likely EXIF "
            "orientation was applied to one and not the other. Feed the segmenter the same image "
            "load_rgb() returned (it normalises orientation) rather than a raw Image.open()."
        )
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((iw, ih), Image.NEAREST)) > 127


def apply_mask_white(image, mask: np.ndarray, fill=WHITE) -> "Image.Image":
    """Replace every non-avocado pixel with white so only the fruit remains — this is the
    actual background removal that matches the light-box's clean white background.

    The mask must already be in the image's coordinates (align_mask_to_image); a mismatch raises
    rather than being resampled here, so a geometry bug surfaces at its source instead of being
    absorbed into a plausible-looking crop.
    """
    from PIL import Image

    arr = np.asarray(image.convert("RGB")).copy()
    if mask.shape != arr.shape[:2]:
        raise ValueError(
            f"mask {mask.shape} does not match image {arr.shape[:2]}; "
            "call align_mask_to_image(mask, image) first"
        )
    arr[~mask] = fill
    return Image.fromarray(arr)


def crop_pad_square(image, bbox: tuple[int, int, int, int], fill=WHITE) -> "Image.Image":
    """Crop to bbox, then paste onto a white square canvas (side = max(w, h)) with the crop
    centered — reproduces the light-box framing (avocado centered on white) without distorting
    aspect ratio."""
    from PIL import Image

    crop = image.crop(bbox)
    w, h = crop.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(crop, ((side - w) // 2, (side - h) // 2))
    return canvas


def preprocess_real_photo(image, segmenter, img_size: int = 224,
                          margin_frac: float = DEFAULT_MARGIN_FRAC,
                          remove_background: bool = True) -> "Image.Image":
    """Full pipeline: segment → erase background to white → crop → white-square pad → resize.

    image: PIL.Image (or path/ndarray). segmenter: any object with .mask(pil) (see load_segmenter).
    Returns a PIL RGB image of (img_size, img_size). Raises NoAvocadoDetected if none is found.
    remove_background: True (default) whites out everything outside the mask — the point of the
    §3 domain-gap fix. Set False to keep the in-bbox background (crop only).
    """
    from PIL import Image

    pil = load_rgb(image)          # display orientation, EXIF tag stripped
    mask = segmenter.mask(pil)
    if mask is None:
        raise NoAvocadoDetected("segmenter found no avocado in the image")
    # One alignment for both consumers below, so the bbox indexes the pixels that got whited out.
    mask = align_mask_to_image(mask, pil)
    if remove_background:
        pil = apply_mask_white(pil, mask)
    square = crop_pad_square(pil, mask_to_bbox(mask, margin_frac))
    return square.resize((img_size, img_size), Image.BILINEAR)


def load_rgb(image) -> "Image.Image":
    """Accept a PIL image, ndarray, or path and return a PIL RGB image in DISPLAY orientation.

    EXIF orientation is applied here and the tag is stripped (see the module docstring): this is
    the single point where every caller — and every segmenter downstream — agrees on which way is
    up. Note that .convert("RGB") alone does NOT drop the tag, so normalising has to be explicit.
    ndarray/tag-less inputs are unaffected (exif_transpose is a no-op without an orientation tag).
    """
    from PIL import Image, ImageOps

    if hasattr(image, "crop"):  # already PIL
        pil = image
    elif isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    else:
        pil = Image.open(image)
    return ImageOps.exif_transpose(pil).convert("RGB")


def _to_ndarray(image) -> np.ndarray:
    """HxWx3 uint8 RGB ndarray (for SAM3, which takes arrays)."""
    return np.asarray(load_rgb(image))


if __name__ == "__main__":
    # Eyeball the background removal on one real photo.
    #   python -m src.preprocess --image my_photos/a.jpg --out out.jpg            # inspyrenet (default)
    #   python -m src.preprocess --image a.jpg --backend rembg                    # fast/light
    #   python -m src.preprocess --image a.jpg --backend sam3 --weights sam3.pt   # SAM3 (needs GPU)
    import argparse

    ap = argparse.ArgumentParser(description="Background-removal preprocessing for one photo")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="preprocessed.jpg")
    ap.add_argument("--backend", default="inspyrenet", choices=["inspyrenet", "rembg", "sam3"])
    ap.add_argument("--mode", default="base", help="inspyrenet backend: base (1024px) | fast (384px)")
    ap.add_argument("--weights", default="sam3.pt", help="sam3 backend: path to gated sam3.pt")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", default=None, help="inspyrenet/sam3 backend: cuda|cpu (auto if unset)")
    ap.add_argument("--keep-background", action="store_true",
                    help="crop only, do NOT white out the background")
    args = ap.parse_args()

    if args.backend == "sam3":
        kw = {"weights": args.weights, "device": args.device}
    elif args.backend == "inspyrenet":
        kw = {"mode": args.mode, "device": args.device}
    else:
        kw = {}
    seg = load_segmenter(args.backend, **kw)
    try:
        result = preprocess_real_photo(     # takes a path directly; load_rgb normalises orientation
            args.image, seg,
            img_size=args.img_size, remove_background=not args.keep_background)
    except NoAvocadoDetected as e:
        raise SystemExit(f"No avocado detected: {e}")
    result.save(args.out)
    print(f"saved -> {args.out}  ({result.size[0]}x{result.size[1]})")
