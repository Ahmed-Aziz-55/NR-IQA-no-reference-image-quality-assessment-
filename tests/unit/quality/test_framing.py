"""
Tests compute_framing_result() — the pure geometry/scoring function —
directly against hand-built synthetic binary masks. Deliberately does
NOT require the u2netp.onnx model file or onnxruntime: ModelFramingDetector's
job is preprocessing + inference + binarization; everything this test
file covers is what happens AFTER that, which is where the actual
framing logic (and its bugs) would live.
"""

import numpy as np
import pytest

from app.quality.semantic.framing import FramingThresholds, compute_framing_result

IMAGE_SIZE = 200  # synthetic masks are IMAGE_SIZE x IMAGE_SIZE for all tests


def _blank_mask() -> np.ndarray:
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)


def _rect_mask(x: int, y: int, w: int, h: int) -> np.ndarray:
    mask = _blank_mask()
    mask[y : y + h, x : x + w] = 255
    return mask


def test_centered_subject_scores_well():
    # A reasonably sized square, centered, well within the frame.
    size = 80
    x = (IMAGE_SIZE - size) // 2
    y = (IMAGE_SIZE - size) // 2
    mask = _rect_mask(x, y, size, size)

    result = compute_framing_result(mask)

    assert result.detected is False
    assert result.touches_edge is False
    assert result.score > 0.7
    assert result.subject_position_score > 0.9
    assert 0.1 < result.subject_area_ratio < 0.2


def test_tiny_region_below_noise_floor_is_discarded_not_flagged():
    # A 10x10 subject (100px / 40000px = 0.25%) is BELOW the default
    # min_component_area_ratio noise floor (1%) — it should be
    # discarded as noise, same as test_noise_only_mask_is_treated_as_empty,
    # not reported as a "detected small subject".
    mask = _rect_mask(95, 95, 10, 10)

    result = compute_framing_result(mask)

    assert result.detected is False
    assert result.subject_area_ratio is None


def test_small_subject_is_penalized():
    # A 22x22 subject (484px / 40000px = 1.21% of the 200x200 frame) —
    # above the noise-filter cutoff (1%) but small enough relative to
    # small_subject_area_ratio (5%) that the size penalty crosses the
    # detected threshold even with perfect centering's small bonus
    # (position contributes only 20% of the score by design — see
    # FramingThresholds.position_weight).
    mask = _rect_mask(89, 89, 22, 22)

    result = compute_framing_result(mask)

    assert result.subject_area_ratio is not None
    assert result.subject_area_ratio < 0.05
    assert result.score < 0.5
    assert result.detected is True


def test_oversized_subject_is_penalized():
    # Subject fills nearly the entire frame (95%+ area) — well above
    # default large_subject_area_ratio (0.85).
    mask = _rect_mask(2, 2, IMAGE_SIZE - 4, IMAGE_SIZE - 4)

    result = compute_framing_result(mask)

    assert result.subject_area_ratio > 0.9
    assert result.score < 1.0
    # Still centered and (barely) not touching the edge at default
    # threshold, so this alone shouldn't necessarily flip `detected`
    # — but the score should clearly reflect the oversized penalty.
    assert result.score < 0.9


def test_edge_touching_subject_is_flagged_as_cropped():
    # Subject's left edge is flush against the image boundary.
    mask = _rect_mask(0, 60, 80, 80)

    result = compute_framing_result(mask)

    assert result.touches_edge is True
    assert result.left_margin == 0.0
    assert result.detected is True


def test_empty_mask_returns_neutral_not_evaluated_result():
    mask = _blank_mask()

    result = compute_framing_result(mask)

    assert result.detected is False
    assert result.score == 0.5
    assert result.touches_edge is None
    assert result.subject_area_ratio is None
    assert result.left_margin is None


def test_noise_only_mask_is_treated_as_empty():
    # Scattered single-pixel noise, no coherent subject — should be
    # filtered out by morphological opening + min_component_area_ratio,
    # same as a genuinely blank mask.
    rng = np.random.default_rng(0)
    mask = _blank_mask()
    noise_points = rng.integers(0, IMAGE_SIZE, size=(50, 2))
    for px, py in noise_points:
        mask[py, px] = 255

    result = compute_framing_result(mask)

    assert result.detected is False
    assert result.subject_area_ratio is None


def test_cluttered_scene_picks_largest_region():
    # Multiple disconnected regions of different sizes (simulating a
    # cluttered background with several objects) — the detector should
    # pick the LARGEST as the candidate subject, consistent with the
    # rest of this project's "largest region = subject" convention
    # (see HeuristicOcclusionDetector, HeuristicFramingDetector).
    mask = _blank_mask()
    mask[10:30, 10:30] = 255  # small region, 20x20 = 400px
    mask[80:140, 80:140] = 255  # large region, 60x60 = 3600px
    mask[170:185, 170:185] = 255  # small region, 15x15 = 225px

    result = compute_framing_result(mask)

    assert result.detected is False
    # The 60x60 region's area ratio: 3600 / (200*200) = 0.09
    assert 0.08 < result.subject_area_ratio < 0.10


def test_custom_thresholds_are_respected():
    # A subject that's "small" under default thresholds should NOT be
    # flagged if a caller supplies a much more permissive threshold.
    mask = _rect_mask(90, 90, 20, 20)  # 400px / 40000px = 1% area ratio
    lenient_thresholds = FramingThresholds(small_subject_area_ratio=0.005)

    default_result = compute_framing_result(mask)
    lenient_result = compute_framing_result(mask, lenient_thresholds)

    assert default_result.score < lenient_result.score


@pytest.mark.parametrize("x,y,w,h", [(0, 0, 50, 50), (150, 150, 50, 50)])
def test_corner_subjects_touch_two_edges(x, y, w, h):
    mask = _rect_mask(x, y, w, h)

    result = compute_framing_result(mask)

    assert result.touches_edge is True