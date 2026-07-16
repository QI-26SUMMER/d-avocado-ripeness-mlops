"""src package. Supports running via `python -m src.<module>`.

Paper reproduction (Foods 2024, DOI 10.3390/foods13081150) pipeline.
Each module uses an import shim (try relative import → except absolute import) so it
works both as `python src/<module>.py` (script) and `python -m src.<module>` (module).
"""
