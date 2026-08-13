"""Ripeness stage constants (CLAUDE.md §1). No dependencies — deliberately.

These are domain facts, not data-loading logic, and three different stages need them:
models.py sizes its classifier head from NUM_CLASSES, serving/app.py names the predicted stage
from LABELS, and data/data.py describes the label column with them. They used to live in
data/data.py, which forced common/models.py to import the data package (a backwards dependency)
and forced the serving image to copy data.py for two constants.

⚠️ Stage 4 is the PEAK (the end of shelf life) and 5 is already past it — do not collapse "4-5"
into one "ripe" bucket. The service's recommended window is stages 3-4.
"""
from __future__ import annotations

LABELS = {1: "Unripe", 2: "Breaking", 3: "Ripe(1)", 4: "Ripe(2)/peak", 5: "Overripe"}
NUM_CLASSES = 5
