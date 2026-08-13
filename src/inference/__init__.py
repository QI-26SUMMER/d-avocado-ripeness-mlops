"""Prediction on real user photos — the deployment path, not the training path.

    preprocess.py  background-removal crop for kitchen photos (CLAUDE.md §3 domain gap).
                   INFERENCE ONLY: training images are already light-box and are left untouched.
                   Also owns load_rgb(), the single point where EXIF orientation is normalised —
                   every user photo must enter the system through it.
    predict.py     CLI: checkpoint -> stage + probability distribution (+ optional shelf-life)

serving/app.py (the Cloud Run container) imports preprocess.py from here; it does not use
predict.py, which is the local CLI.
"""
