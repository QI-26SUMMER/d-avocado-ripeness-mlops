"""Image preprocessing/augmentation transforms (paper reproduction §5, §6).

Clearly separates three transforms:
  - paper_train_transform      : only the paper's geometric augmentations (reflection/rotation/rescale/translation)
  - evaluation_transform       : no augmentation (val/test only)
  - baseline_noaug_transform   : preserves the original no-augmentation baseline (§14 B0 preservation)

Paper (Foods 2024) augmentation spec:
  random reflection, rotation -10 deg to +10 deg, rescaling 0.95-1.05,
  horizontal/vertical translation -10 to +10 pixels. (No color/brightness augmentation included)
The paper used MATLAB's imageDataAugmenter -> implemented here with the closest PyTorch
equivalent, RandomAffine/Flip (an implementation choice).
"""
from __future__ import annotations

from torchvision import transforms

# ImageNet pretrained normalization (required for transfer learning)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize():
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def evaluation_transform(img_size: int = 224):
    """val/test only. Absolutely no augmentation (§6)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        _normalize(),
    ])


# Identical to the original no-augmentation baseline (§14 B0 preservation)
baseline_noaug_transform = evaluation_transform


def paper_train_transform(img_size: int = 224, translate_px: int = 10):
    """Applies only the paper's geometric augmentations (train only).

    translate: +-10px -> RandomAffine's translate is a fraction, so translate_px/img_size.
    rotation +-10 deg, scale 0.95-1.05, reflection = H/V flip.
    RandomAffine is applied before ToTensor (i.e. at the PIL stage), filling with a
    white background (fill=255) to match the light box.
    """
    frac = translate_px / img_size
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),   # random reflection
        transforms.RandomVerticalFlip(p=0.5),     # random reflection
        transforms.RandomAffine(
            degrees=10,                            # rotation -10 deg to +10 deg
            translate=(frac, frac),                # H/V translation +-10px
            scale=(0.95, 1.05),                    # rescaling
            fill=255,                              # fill with matte white background
        ),
        transforms.ToTensor(),
        _normalize(),
    ])


def build_train_transform(kind: str, img_size: int = 224):
    """Select the train transform based on the augmentation kind from config.

    kind: 'paper' -> paper_train_transform, 'none' -> no augmentation.
    """
    if kind == "paper":
        return paper_train_transform(img_size)
    if kind == "none":
        return evaluation_transform(img_size)
    raise ValueError(f"unknown augmentation kind: {kind!r} (paper|none)")
