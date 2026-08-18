from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.quality.classical._image_utils import MIN_DIMENSION, to_grayscale, validate_image

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised only when onnxruntime is missing
    ort = None


@dataclass(frozen=True)
class FramingResult:
    """
    Result produced by a framing detector.

    Attributes:
        score: Framing quality score from 0.0 (poorly framed) to
            1.0 (well framed).
        detected: Whether significant framing issues were detected.
        touches_edge: Whether the subject appears to be cropped by
            the image boundary. None if not evaluated/unknown.
        subject_position_score: How centered/well-positioned the
            subject is, from 0.0 (poorly positioned) to 1.0 (well
            positioned). None if not evaluated/unknown. This is a
            SOFT signal, not a hard rule — an off-center subject is
            not automatically bad framing (compositional choice), so
            it contributes only a minor weight to `score` — see
            ModelFramingDetector's docstring.
        subject_area_ratio: Fraction of the image the detected
            subject occupies (0.0-1.0). None if not evaluated/unknown.
            Very low = subject lost in the frame; very high = subject
            dominates/likely over-cropped.
        left_margin: Normalized (0.0-1.0) empty space between the
            subject's bounding box and the image's left edge. None if
            not evaluated/unknown.
        right_margin: Same, for the right edge.
        top_margin: Same, for the top edge.
        bottom_margin: Same, for the bottom edge.
    """

    score: float
    detected: bool
    touches_edge: bool | None = None
    subject_position_score: float | None = None
    subject_area_ratio: float | None = None
    left_margin: float | None = None
    right_margin: float | None = None
    top_margin: float | None = None
    bottom_margin: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"score must be between 0.0 and 1.0, got {self.score}."
            )

        for field_name in (
            "subject_position_score",
            "subject_area_ratio",
            "left_margin",
            "right_margin",
            "top_margin",
            "bottom_margin",
        ):
            value = getattr(self, field_name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}.")


class FramingDetector(ABC):
    """
    Interface for semantic framing quality detection.
    """

    @abstractmethod
    def assess(self, image: np.ndarray) -> FramingResult:
        """
        Assess an image for framing quality.

        Args:
            image: Input image as a NumPy array.

        Returns:
            FramingResult containing the quality score and supporting
            signals (edge-touching, subject position).
        """
        raise NotImplementedError


class HeuristicFramingDetector(FramingDetector):
    """
    Classical, no-training heuristic framing detector.

    **Kept as a dependency-free fallback / test double now that
    ModelFramingDetector is the primary implementation.** Left
    unmodified and NOT deleted — per project decision, the model
    detector is being integrated as an alternative, not a hard
    replacement, until evaluation on real data justifies retiring
    this one.

    This heuristic has no concept of "subject" — it approximates one
    as the largest foreground contour found via edge detection, which
    will misfire whenever the largest contour isn't the actual subject
    (cluttered scenes, low-contrast subjects, multiple similarly sized
    objects). See ModelFramingDetector for the saliency-based approach
    that addresses this.

    Method:
      1. Grayscale + Canny edge detection, dilated to close gaps.
      2. Take the largest contour's bounding box as the candidate subject.
      3. touches_edge: True if that bounding box touches any image
         boundary (subject likely cropped).
      4. subject_position_score: 1.0 minus the normalized distance
         between the bounding box center and the image center (closer
         to center = higher score).
      5. score: combines edge-touching penalty and position score.

    If no contour is found (e.g. a blank/flat image), the detector
    reports detected=False with a neutral score, since "no discernible
    subject" isn't the same failure mode as "subject poorly framed".
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        edge_touch_penalty: float = 0.4,
    ) -> None:
        """
        Args:
            canny_low: Lower threshold for Canny edge detection.
            canny_high: Upper threshold for Canny edge detection.
            edge_touch_penalty: Score deducted when the candidate subject
                touches an image edge. Not calibrated against labeled
                data yet.
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.edge_touch_penalty = edge_touch_penalty

    def _largest_contour_bbox(self, gray: np.ndarray) -> tuple[int, int, int, int] | None:
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 1.0:
            return None

        return cv2.boundingRect(largest)  # (x, y, w, h)

    def assess(self, image: np.ndarray) -> FramingResult:
        validate_image(image, min_dimension=MIN_DIMENSION)
        gray = to_grayscale(image)
        height, width = gray.shape

        bbox = self._largest_contour_bbox(gray)
        if bbox is None:
            return FramingResult(score=0.5, detected=False)

        x, y, w, h = bbox
        margin = 2  # pixels of tolerance for "touching" the boundary
        touches_edge = (
            x <= margin
            or y <= margin
            or (x + w) >= (width - margin)
            or (y + h) >= (height - margin)
        )

        bbox_center_x = x + w / 2
        bbox_center_y = y + h / 2
        image_center_x = width / 2
        image_center_y = height / 2

        max_dist = np.hypot(image_center_x, image_center_y)
        actual_dist = np.hypot(
            bbox_center_x - image_center_x, bbox_center_y - image_center_y
        )
        subject_position_score = float(
            np.clip(1.0 - (actual_dist / max_dist if max_dist > 0 else 0.0), 0.0, 1.0)
        )

        score = subject_position_score
        if touches_edge:
            score = max(0.0, score - self.edge_touch_penalty)

        detected = touches_edge or subject_position_score < 0.5

        return FramingResult(
            score=float(score),
            detected=detected,
            touches_edge=touches_edge,
            subject_position_score=subject_position_score,
        )


@dataclass(frozen=True)
class FramingThresholds:
    """
    All ModelFramingDetector thresholds/weights in one place, so
    nothing is scattered as inline magic numbers and everything is
    easy to recalibrate later against labeled validation images.

    NONE of these are calibrated against labeled data yet — they are
    reasonable starting points only. Recalibrate once real framing
    labels are available (see docs/decision_engine.md convention used
    elsewhere in this project for other uncalibrated thresholds).
    """

    # Saliency mask binarization: a pixel counts as "subject" if the
    # model's per-pixel saliency score exceeds this.
    mask_threshold: float = 0.5

    # Denoising: connected components smaller than this fraction of
    # total image area are treated as noise, not a candidate subject.
    min_component_area_ratio: float = 0.01

    # Morphological opening kernel (pixels) applied before connected
    # components, to remove small disconnected speckle noise in the
    # binarized mask.
    morph_kernel_size: int = 5

    # A subject bounding-box edge closer than this fraction of the
    # corresponding image dimension to the image boundary counts as
    # "touching the edge" (likely cropped). Initial guess, not
    # calibrated — tune against real cropped/uncropped examples.
    edge_margin_threshold: float = 0.02

    # subject_area_ratio below this = subject is small enough to be
    # considered "lost in the frame".
    small_subject_area_ratio: float = 0.05

    # subject_area_ratio above this = subject dominates the frame
    # enough to be considered likely over-cropped.
    large_subject_area_ratio: float = 0.85

    # Flat score deduction applied when touches_edge is True.
    edge_touch_penalty: float = 0.3

    # How much subject_position_score contributes to the final score.
    # Deliberately small — an off-center subject is a compositional
    # choice, not inherently bad framing, so this is a soft nudge
    # rather than a hard "center = good" rule.
    position_weight: float = 0.2

    # score below this (in addition to touches_edge) counts as
    # "framing issue detected".
    detected_score_threshold: float = 0.5


DEFAULT_FRAMING_THRESHOLDS = FramingThresholds()


@dataclass(frozen=True)
class _SubjectRegion:
    """Internal: the largest valid connected component in a binarized mask."""

    x: int
    y: int
    w: int
    h: int
    pixel_area_ratio: float


def _largest_valid_component(
    binary_mask: np.ndarray,
    thresholds: FramingThresholds,
) -> _SubjectRegion | None:
    """
    Finds the largest connected foreground component in a binary mask,
    after morphological opening to strip small speckle noise. Returns
    None if no component survives the min_component_area_ratio cutoff
    (i.e. no discernible subject — an empty/near-empty saliency mask).
    """
    height, width = binary_mask.shape[:2]
    total_pixels = height * width

    kernel = np.ones((thresholds.morph_kernel_size, thresholds.morph_kernel_size), np.uint8)
    denoised = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(denoised, connectivity=8)
    if num_labels <= 1:
        return None

    # Label 0 is background; evaluate the rest.
    best_label = None
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label

    if best_label is None:
        return None

    area_ratio = best_area / total_pixels
    if area_ratio < thresholds.min_component_area_ratio:
        return None

    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])

    return _SubjectRegion(x=x, y=y, w=w, h=h, pixel_area_ratio=area_ratio)


def compute_framing_result(
    binary_mask: np.ndarray,
    thresholds: FramingThresholds = DEFAULT_FRAMING_THRESHOLDS,
) -> FramingResult:
    """
    Pure geometry/scoring function: takes an already-binarized (0/255
    uint8) subject mask and turns it into a FramingResult. Deliberately
    separated from ONNX inference (see ModelFramingDetector) so this
    logic is unit-testable with hand-built synthetic masks, no model
    file required.

    Args:
        binary_mask: 2D uint8 array, same height/width as the source
            image, values 0 or 255 — the thresholded saliency mask.
        thresholds: All tunable cutoffs/weights — see FramingThresholds.

    Returns:
        FramingResult. If no valid subject region survives denoising
        (empty/near-empty mask), returns a neutral "not evaluated"
        result (detected=False, score=0.5, all subject-specific
        fields None) — mirrors HeuristicFramingDetector's handling of
        "no contour found".
    """
    height, width = binary_mask.shape[:2]

    region = _largest_valid_component(binary_mask, thresholds)
    if region is None:
        return FramingResult(score=0.5, detected=False)

    left_margin = region.x / width
    right_margin = (width - (region.x + region.w)) / width
    top_margin = region.y / height
    bottom_margin = (height - (region.y + region.h)) / height

    touches_edge = min(left_margin, right_margin, top_margin, bottom_margin) < thresholds.edge_margin_threshold

    # Size score: 1.0 within the "normal" range, penalized smoothly
    # outside it in either direction.
    area_ratio = region.pixel_area_ratio
    if area_ratio < thresholds.small_subject_area_ratio:
        size_score = area_ratio / thresholds.small_subject_area_ratio
    elif area_ratio > thresholds.large_subject_area_ratio:
        remaining_range = max(1e-6, 1.0 - thresholds.large_subject_area_ratio)
        size_score = 1.0 - (area_ratio - thresholds.large_subject_area_ratio) / remaining_range
    else:
        size_score = 1.0
    size_score = float(np.clip(size_score, 0.0, 1.0))

    bbox_center_x = region.x + region.w / 2
    bbox_center_y = region.y + region.h / 2
    image_center_x = width / 2
    image_center_y = height / 2
    max_dist = np.hypot(image_center_x, image_center_y)
    actual_dist = np.hypot(bbox_center_x - image_center_x, bbox_center_y - image_center_y)
    subject_position_score = float(
        np.clip(1.0 - (actual_dist / max_dist if max_dist > 0 else 0.0), 0.0, 1.0)
    )

    # Position is a soft signal (see FramingThresholds.position_weight
    # docstring) — an off-center subject alone shouldn't tank the score.
    score = (1 - thresholds.position_weight) * size_score + thresholds.position_weight * subject_position_score
    if touches_edge:
        score = max(0.0, score - thresholds.edge_touch_penalty)
    score = float(np.clip(score, 0.0, 1.0))

    detected = touches_edge or score < thresholds.detected_score_threshold

    return FramingResult(
        score=score,
        detected=detected,
        touches_edge=touches_edge,
        subject_position_score=subject_position_score,
        subject_area_ratio=float(np.clip(area_ratio, 0.0, 1.0)),
        left_margin=float(np.clip(left_margin, 0.0, 1.0)),
        right_margin=float(np.clip(right_margin, 0.0, 1.0)),
        top_margin=float(np.clip(top_margin, 0.0, 1.0)),
        bottom_margin=float(np.clip(bottom_margin, 0.0, 1.0)),
    )


class ModelFramingDetector(FramingDetector):
    """
    Saliency-based framing detector using U²-NetP (Qin et al., "U²-Net:
    Going Deeper with Nested U-Structure for Salient Object Detection",
    Pattern Recognition 2020; ONNX export via Heliosoph/u2net-onnx on
    HuggingFace, Apache-2.0).

    IMPORTANT SCOPING NOTE (matches the project's occlusion detector
    precedent): U²-Net provides SUBJECT LOCALIZATION, not a framing
    verdict by itself. It finds the most visually salient region of an
    image — this is usually the intended subject (a person, product,
    animal, vehicle), but not always: landscapes, cluttered scenes, and
    group shots can produce a salient region that isn't "the subject"
    a human would name. So the architecture here is deliberately:

        U²-NetP saliency mask -> geometric measurements -> FramingResult

    NOT "U²-Net directly outputs a framing quality score". All actual
    scoring logic lives in compute_framing_result() (a pure function,
    unit-tested independently of ONNX inference) — this class only
    handles preprocessing, inference, and mask binarization/denoising.

    Like ModelOcclusionDetector, this is being integrated as an
    ALTERNATIVE alongside HeuristicFramingDetector, not a replacement —
    the heuristic stays in the codebase until evaluation on real data
    justifies retiring it.
    """

    INPUT_SIZE = 320
    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        onnx_model_path: str | Path,
        thresholds: FramingThresholds = DEFAULT_FRAMING_THRESHOLDS,
    ) -> None:
        """
        Args:
            onnx_model_path: Path to u2netp.onnx. Input: [1,3,320,320]
                float32 RGB, scaled to [0,1] then ImageNet-normalized,
                NCHW. Output: 7 tensors (d0..d6); d0 (index 0) is the
                final fused saliency map, [1,1,320,320], per-pixel
                score in roughly [0,1] (min-max normalized here to be
                safe, since U²-Net's raw output isn't strictly bounded).
            thresholds: See FramingThresholds — every tunable cutoff.

        Raises:
            ImportError: If onnxruntime is not installed.
            FileNotFoundError: If the ONNX model can't be loaded.
        """
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ModelFramingDetector. "
                "Install it with `pip install onnxruntime`."
            )

        onnx_model_path = Path(onnx_model_path)
        if not onnx_model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

        self.thresholds = thresholds
        self._session = ort.InferenceSession(
            str(onnx_model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            image_rgb, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR
        )
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self._IMAGENET_MEAN) / self._IMAGENET_STD
        chw = normalized.transpose(2, 0, 1)
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def _predict_saliency_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Returns a saliency mask resized to the original image dimensions, values in [0, 1]."""
        height, width = image_bgr.shape[:2]
        input_tensor = self._preprocess(image_bgr)

        outputs = self._session.run(None, {self._input_name: input_tensor})
        d0 = outputs[0][0, 0]  # final fused mask, 320x320

        # U²-Net's raw output isn't strictly bounded to [0,1] — min-max
        # normalize defensively before thresholding.
        d0_min, d0_max = float(d0.min()), float(d0.max())
        if d0_max - d0_min > 1e-6:
            d0 = (d0 - d0_min) / (d0_max - d0_min)
        else:
            # Degenerate case: a perfectly flat saliency map (e.g. a
            # blank/uniform input) — nothing is salient.
            d0 = np.zeros_like(d0)

        return cv2.resize(d0, (width, height), interpolation=cv2.INTER_LINEAR)

    def assess(self, image: np.ndarray) -> FramingResult:
        validate_image(image, min_dimension=MIN_DIMENSION)

        if image.ndim != 3:
            raise ValueError(
                "ModelFramingDetector requires a color (BGR/BGRA) image; "
                "grayscale input is not supported."
            )

        image_bgr = image[:, :, :3] if image.shape[2] == 4 else image

        saliency = self._predict_saliency_mask(image_bgr)
        binary_mask = (saliency > self.thresholds.mask_threshold).astype(np.uint8) * 255

        return compute_framing_result(binary_mask, self.thresholds)