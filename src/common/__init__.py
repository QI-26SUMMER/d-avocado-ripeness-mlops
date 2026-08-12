"""Shared across every pipeline stage — imported by data/, training/ and inference/ alike.

Nothing here may import from data/, training/ or inference/: the dependency arrow points one
way only. That is why the label constants live in labels.py rather than in data/data.py, where
they used to force common/models.py to import the data package for NUM_CLASSES.

    labels.py       ripeness stage names + NUM_CLASSES (domain constants, no dependencies)
    utils.py        seeding, config loading, OUTPUT_DIR, experiment logging
    metrics.py      ordinal 5-stage metric bundle (accuracy, MAE, within-1, QWK, ...)
    models.py       ResNet-18 / AlexNet backbones
    transforms.py   train/eval image transforms
    shelf_life.py   days_left (paper) + days_to_target (serving), temperature -> alpha

The serving container copies only labels.py, models.py, transforms.py and shelf_life.py from
here, so keep those four free of pandas and other training-only dependencies.
"""
