"""Tests for the serving response shape after the backend field-name alignment.

Covers only `_predict_one` in serving/app.py: verifies the response uses the
Spring backend's DB column names (predicted_stage, stage_probs, model_version)
instead of the old (stage, probs) names, and that days_left is left untouched.
Uses a stubbed model + transform via the module-level `_state` dict, so no
real checkpoint, GCS access, or Vertex AI plumbing is needed.

Run (either works):
  python -m pytest tests/test_serving_response.py -q
  python tests/test_serving_response.py
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # for `import serving.app` (namespace package, no __init__.py)
sys.path.insert(0, str(ROOT / "src"))  # serving/app.py itself expects src/ on sys.path

from PIL import Image  # noqa: E402

from serving.app import _predict_one, _state  # noqa: E402


def _make_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (8, 8), (120, 150, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class _StubModel:
    """Ignores its input and always returns the same fixed logits (stage 3 wins)."""

    def __call__(self, x):
        import torch

        return torch.tensor([[0.0, 0.5, 4.0, 1.0, 0.0]])


def _set_stub_state(experiment_id: str = "TEST_experiment_v1") -> dict:
    """Monkeypatches the module-level _state with a stub model/transform.

    Returns the original _state contents so the caller can restore them.
    """
    import torch

    original = dict(_state)
    _state["model"] = _StubModel()
    _state["transform"] = lambda img: torch.zeros(3, 4, 4)
    _state["device"] = "cpu"
    _state["cfg"] = {"experiment_id": experiment_id}
    return original


def _restore_state(original: dict) -> None:
    _state.clear()
    _state.update(original)


def test_response_has_backend_contract_keys():
    original = _set_stub_state()
    try:
        out = _predict_one(_make_jpeg_bytes(), storage_group=None)
        assert set(out.keys()) == {
            "predicted_stage", "stage_probs", "confidence", "model_version", "label", "hint",
        }
    finally:
        _restore_state(original)


def test_stage_probs_is_a_5way_distribution_matching_confidence():
    original = _set_stub_state()
    try:
        out = _predict_one(_make_jpeg_bytes(), storage_group=None)
        probs = out["stage_probs"]
        assert isinstance(probs, list) and len(probs) == 5
        assert all(isinstance(p, float) for p in probs)
        assert abs(sum(probs) - 1.0) < 1e-6
        assert probs[out["predicted_stage"] - 1] == out["confidence"]
    finally:
        _restore_state(original)


def test_model_version_matches_stubbed_cfg():
    original = _set_stub_state(experiment_id="P1_general_resnet18_paper_aug_oversample")
    try:
        out = _predict_one(_make_jpeg_bytes(), storage_group=None)
        assert out["model_version"] == "P1_general_resnet18_paper_aug_oversample"
    finally:
        _restore_state(original)


def test_days_left_only_present_with_storage_group():
    original = _set_stub_state()
    try:
        out_no_group = _predict_one(_make_jpeg_bytes(), storage_group=None)
        assert "days_left" not in out_no_group

        out_with_group = _predict_one(_make_jpeg_bytes(), storage_group="T20")
        assert "days_left" in out_with_group
    finally:
        _restore_state(original)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
