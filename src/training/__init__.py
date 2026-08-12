"""Training and evaluation — everything that needs the labelled dataset on disk.

    train.py         config-driven training loop (single fixed split)
    cv_train.py      group-aware k-fold cross-validation (resumable)
    evaluate.py      test-set analysis: image-level / best-side / two-side + shelf-life
    sampler.py       paper-style random oversampling (train split only)
    gmm_baseline.py  GMM colour baseline, the classical-ML comparison point for ResNet

Requires src.data to have been run first (metadata_clean.csv + splits.csv). Nothing in the
serving container imports this package.
"""
