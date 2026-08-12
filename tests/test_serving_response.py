"""Tests for the serving response shape after the backend field-name alignment.

Covers `_predict_one` in serving/app.py (backend contract keys, days_to_target)
and `alpha_from_temp` in src/common/shelf_life.py (Q10 log-linear interpolation).
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
from common.shelf_life import alpha_from_temp  # noqa: E402


def _make_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (8, 8), (120, 150, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class _StubModel:
    """Ignores its input and always returns the same fixed logits (stage wins)."""

    def __init__(self, stage: int = 3):
        self.stage = stage

    def __call__(self, x):
        import torch

        logits = [0.0] * 5
        logits[self.stage - 1] = 4.0
        return torch.tensor([logits])


def _set_stub_state(experiment_id: str = "TEST_experiment_v1", stage: int = 3) -> dict:
    """Monkeypatches the module-level _state with a stub model/transform.

    Returns the original _state contents so the caller can restore them.
    """
    import torch

    original = dict(_state)
    _state["model"] = _StubModel(stage)
    _state["transform"] = lambda img: torch.zeros(3, 4, 4)
    _state["device"] = "cpu"
    _state["cfg"] = {"experiment_id": experiment_id}
    return original


def _restore_state(original: dict) -> None:
    _state.clear()
    _state.update(original)


def test_alpha_from_temp_anchors_and_interpolation():
    assert round(alpha_from_temp(10), 2) == -4.0
    assert round(alpha_from_temp(20), 2) == -1.75
    assert round(alpha_from_temp(17), 2) == -2.24


def test_alpha_from_temp_flat_above_20():
    # Ripening rate plateaus above 20°C (separate paper's final-ripening-temp data),
    # so alpha stops decreasing past the 20°C anchor instead of extrapolating further.
    at_20 = alpha_from_temp(20)
    assert alpha_from_temp(23) == at_20
    assert alpha_from_temp(25) == at_20


def test_alpha_from_temp_clamps_above_25():
    # Accepted input range tops out at 25°C (TEMP_CLAMP); higher inputs are
    # clamped rather than extrapolated, so they still land on the 20°C anchor.
    assert alpha_from_temp(30) == alpha_from_temp(25)


def test_response_has_backend_contract_keys():
    original = _set_stub_state()
    try:
        out = _predict_one(_make_jpeg_bytes())
        assert set(out.keys()) == {
            "predicted_stage", "stage_probs", "confidence", "model_version", "label", "hint",
        }
    finally:
        _restore_state(original)


def test_stage_probs_is_a_5way_distribution_matching_confidence():
    original = _set_stub_state()
    try:
        out = _predict_one(_make_jpeg_bytes())
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
        out = _predict_one(_make_jpeg_bytes())
        assert out["model_version"] == "P1_general_resnet18_paper_aug_oversample"
    finally:
        _restore_state(original)


def test_days_to_target_only_present_with_target_stage():
    original = _set_stub_state(stage=2)
    try:
        out_no_target = _predict_one(_make_jpeg_bytes())
        assert "days_to_target" not in out_no_target

        out_with_target = _predict_one(_make_jpeg_bytes(), target_stage=4, temp_celsius=17.0)
        assert "days_to_target" in out_with_target
    finally:
        _restore_state(original)


def test_days_to_target_value_at_17c():
    # predicted stage 2, target 4, alpha_from_temp(17) ~= -2.24 -> -2.24*(2-4) ~= 4.5
    original = _set_stub_state(stage=2)
    try:
        out = _predict_one(_make_jpeg_bytes(), target_stage=4, temp_celsius=17.0)
        assert out["days_to_target"] == 4.5
        assert isinstance(out["days_to_target"], float)
    finally:
        _restore_state(original)


def test_days_to_target_negative_when_overripe():
    # predicted stage 5, target 4 -> already past target -> negative, not clipped to 0
    original = _set_stub_state(stage=5)
    try:
        out = _predict_one(_make_jpeg_bytes(), target_stage=4, temp_celsius=17.0)
        assert out["days_to_target"] < 0
    finally:
        _restore_state(original)


def test_no_target_stage_does_not_crash():
    original = _set_stub_state()
    try:
        out = _predict_one(_make_jpeg_bytes(), target_stage=None)
        assert "days_to_target" not in out
        assert "predicted_stage" in out
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
