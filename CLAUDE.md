CLAUDE.md — Avocado Ripeness Model

The operating manual Claude Code reads when working in this repository. Current stage: data analysis.
Once the training stage begins, training-related instructions will be added.


Attitude: neural network training fails silently. No error — it just quietly hands you a slightly worse model.
So validate paranoidly, and change only one thing at a time.




0. Context

Goal. One RGB photo → (a) ripeness stage classification, (b) estimate of days remaining until peak edibility.
The final deliverable is a working service, not a paper.

Data. Hass Avocado Ripening Photographic Dataset (Mendeley, CC BY 4.0, DOI 10.17632/3xd9n945v8.1)

Baseline. Original paper (Foods, 2024): AlexNet / ResNet-18 transfer learning, 88.8% accuracy,
96.7% of predictions within half a stage of error.
⚠️ This figure is measured on each sample's "best side." Do not compare it directly against single-image accuracy.


1. Verified data facts

Don't guess — use these. Values confirmed via EDA.

ItemValueImages on disk14,710 (.jpg, 800×800, approx. 471.5 MB)Metadata rows14,722 (Avocado Ripening Dataset.xlsx, sheet DATABASE)Missing images12 (present in metadata but file missing)Individuals (Sample) count478Photographed sidesa 7,361 / b 7,361

Storage groups

GroupImagesSamplesObservation daysConditionT108,870 (60.3%)192Day 1–2610°C / 85% RHT202,926 (19.9%)143Day 1–1120°C / 85% RHTam2,926 (19.9%)143Day 1–11Ambient (room temperature)

Label distribution

Label12345Count3,5722,2342,7583,2942,864Percentage24.3%15.2%18.7%22.4%19.5%

Label meanings — understand these precisely

1 Unripe    Unripe (yellow-green, very firm)
2 Breaking  Transition stage (grayish-olive)
3 Ripe(1)   Edible (purple spotting)
4 Ripe(2)   Peak ← the end point of shelf life
5 Overripe  Past its peak ← not the service target

⚠️ 4 is the peak; 5 is already too late. Do not lump "4-5 = fully ripe" together. The recommended window is stages 3-4.

Filename convention

T{group}_d{day:02d}_{sample:03d}_{a|b}_{label}.jpg
e.g., T10_d09_072_a_2.jpg

Verified that the filenames and Excel columns match exactly across all 14,722 records.

Watch for nested folders

data/Hass Avocado Ripening Photographic Dataset/Hass Avocado Ripening Photographic Dataset/...


2. 🔴 Pitfalls — this is where most people fail

2.1 Split by individual/sample, not by image ❌

The same avocado appears in an average of 30 images (roughly 15 days × two sides). A random image-level split
puts the same individual in both train and test — the model ends up memorizing that avocado's blemishes instead of its ripeness.

pythonfrom sklearn.model_selection import GroupShuffleSplit
groups = df["Storage Group"] + "_" + df["Sample"].astype(str)

After splitting, verify: set(train_groups) & set(test_groups) == set()

2.2 Verify that Sample numbers are globally unique

T20 and Tam both have exactly 143 samples / 2,926 images. Don't trust it until you've checked it yourself.
If numbering restarts at 1 within each group, groupby("Sample") will merge different fruit into a single group.

pythonassert df.groupby("Sample")["Storage Group"].nunique().max() == 1, \
    "Sample IDs are reused → group key must be the (Storage Group, Sample) tuple"

2.3 Do not use Storage Group as a model input

This information does not exist at inference time. The model has no way of knowing whether a user's photographed avocado is T10 or Tam,
and at the same ripeness stage the appearance is the same regardless. Including it only inflates validation performance and collapses in the live service.

Image                  → current ripeness stage  (CNN)
Stage + storage method → days remaining           (storage method is a value the UI asks the user for)

2.4 Labels are ordinal

Confusing 1↔5 is worse than confusing 1↔2. Don't stop at accuracy alone —
report MAE, within-half-stage, and QWK (quadratic weighted kappa) together.

2.5 Labels may have temporal context baked in

478/478 samples are 100% monotonically increasing. If humans had labeled the photos independently, you'd expect
about a day's worth of reversal at the boundaries (2↔3, 3↔4). 100% is a sign that the annotator enforced monotonicity while looking at the time series.

Our model only ever sees a single photo. → There may be a performance ceiling for a single-image model on boundary samples.
Don't jump to suspecting your code just because you can't beat 88.8%.
Related prior work: Hybrid Loss for Robust Ripeness Classification under Noisy Labels (on this same dataset).

2.6 days_left is not in the dataset — it must be derived

The Excel columns are only File Name / Time Stamp / Storage Group / Sample / Day of Experiment / Ripening Index Classification.

pythonend = (df[df["Ripening Index Classification"] == 5]
       .groupby(["Storage Group", "Sample"])["Day of Experiment"].min())

⚠️ Right-censoring. T10's mean ripening index on day 26 is 4.98, not 5.00.
Some T10 individuals never reached label 5 by the end of the experiment → days_left is undefined for them.
Simply dropping them creates a selection bias that discards "slow-ripening individuals." Keep a censoring flag instead.

2.7 Do not compare groups using calendar dates

The shooting period runs 2022-04-03 to 2022-05-12 (39 days), yet the maximum Day of Experiment is 26.
Each group starts on a different date (they're separate cohorts). Always use Day of Experiment as the time axis.

2.8 Unverified claims — do not cite these


❌ "There is a 10-stage ripening index version" → there are only 5 stages
❌ "The paper used a 70/15/15 split" → not confirmed in the original text
❌ "days_left is provided as a label" → it is not provided



3. Known limitations (to be documented, not solved right now)

Domain gap. Training images were shot in a lightbox (matte white background, 12,000 lm LED, 5500K, fixed exposure).
Real users will shoot with a phone under kitchen lighting. Ripeness is judged by color, and lighting changes color.

→ Apply aggressive color/lighting augmentation during training.
→ Do not build preprocessing that depends on background uniformity (e.g., simple threshold-based segmentation).


4. Evaluation protocol (fixed)

Split      : GroupShuffleSplit on (Storage Group, Sample)   — not image-level ❌
Metrics    : accuracy, MAE, within-half-stage, QWK, per-class recall
Report unit: (1) single image  (2) best-side  — report both

If validation accuracy suddenly exceeds 95%, suspect a leak before you celebrate.


5. Repository conventions

repo/
├── CLAUDE.md
├── data/
│   ├── .gitignore                        # excludes all of images/
│   └── Avocado Ripening Dataset.xlsx     # ✅ committed (small and essential)
├── docs/
│   ├── eda.md
│   └── foods-2024-avocado.pdf            # read only when needed
├── notebooks/
└── src/
    ├── data.py      # loader, filters out the 12 missing images, group split
    ├── models.py
    ├── train.py
    └── evaluate.py


Do not commit images to git (471 MB). Provide a script that downloads them via DOI instead.
Fix the seed. A result that isn't reproducible isn't a result.
Leave performance numbers in commit messages: feat(model): resnet18 baseline, val acc 0.81 / MAE 0.24



6. Citation

CC BY 4.0 — attribution is a condition of the license.

Dataset: 'Hass' Avocado Ripening Photographic Dataset, Mendeley Data.
DOI: 10.17632/3xd9n945v8.1

Paper: Shelf-Life Management and Ripening Assessment of 'Hass' Avocado
(Persea americana) Using Deep Learning Approaches. Foods, 2024.

Fill in author names by checking the original source directly.


7. To Claude Code


Do not write code that violates §2, especially 2.1 (split by individual) and 2.3 (group leakage).
If you're not sure, ask instead of guessing. §2.8 already lists facts that turned out to be wrong once.
Keep code short and explicit. This repository is maintained by a single developer.