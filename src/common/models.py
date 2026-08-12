"""Model definitions — paper reproduction §8 (ResNet-18 / AlexNet, ImageNet pretrained).

Replaces the final classification layer of both pretrained backbones with 5 classes.
The paper used MATLAB AlexNet (227×227 input), but here we use torchvision AlexNet
(standard 224×224) — input size is an implementation choice, specified via config's
img_size (§5).
"""
from __future__ import annotations

import torch.nn as nn

try:  # relative when imported as src.common.models; absolute when src/ is on sys.path
    from .labels import NUM_CLASSES
except ImportError:
    from common.labels import NUM_CLASSES


def resnet18_baseline(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ResNet-18, ImageNet pretrained. Replaces the final fc layer with 5 classes."""
    import torchvision

    weights = "IMAGENET1K_V1" if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def alexnet_baseline(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """AlexNet, ImageNet pretrained. Replaces the classifier's final Linear layer with 5 classes.

    torchvision AlexNet standard input=224×224 (the paper's MATLAB version was 227×227 —
    implementation difference, noted in the report).
    """
    import torchvision

    weights = "IMAGENET1K_V1" if pretrained else None
    model = torchvision.models.alexnet(weights=weights)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Selects backbone via config's model.name."""
    name = name.lower()
    if name in {"resnet18", "resnet-18"}:
        return resnet18_baseline(num_classes, pretrained)
    if name in {"alexnet"}:
        return alexnet_baseline(num_classes, pretrained)
    raise ValueError(f"unknown model: {name!r} (resnet18|alexnet)")


# Improvement experiments (kept separate from paper reproduction): ordinal head (CORAL/CORN), noisy-label handling, etc.
