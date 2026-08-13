"""src package. Supports running via `python -m src.<package>.<module>`.

Paper reproduction (Foods 2024, DOI 10.3390/foods13081150) pipeline, split by pipeline stage:

    common/     shared by every stage (labels, utils, metrics, models, transforms, shelf_life)
    data/       dataset preparation      -> python -m src.data.validate_data ; src.data.split
    training/   training and evaluation  -> python -m src.training.train ; src.training.evaluate
    inference/  real-photo prediction    -> python -m src.inference.predict

Dependencies point one way: data/training/inference all import from common/, never the reverse.
Each module uses an import shim (try relative import → except absolute import) so it works both
as `python -m src.<package>.<module>` and with src/ on sys.path (what tests and serving/app.py
do), which is why nothing here imports via the `src.` prefix.
"""
