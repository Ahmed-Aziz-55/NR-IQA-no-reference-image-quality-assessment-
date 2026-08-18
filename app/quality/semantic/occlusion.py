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
class OcclusionResult:
    """
    Result produced by an occlusion detector.

    Attributes:
        score: Occlusion quality score from 0.0 (fully occluded) to
            1.0 (no occlusion).
        detected: Whether significant occlusion was detected.
        face_detected: Whether a face was found in the image before
            scoring. True/False for detectors that face-gate (see
            ModelOcclusionDetector); None for detectors that don't do
            face-gating at all (e.g. the legacy heuristic below).
            False means occlusion was NOT evaluated because no face
            was found — score/detected are neutral placeholders in
            that case, not a real verdict.
    """

    score: float
    detected: bool
    face_detected: bool | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"score must be between 0.0 and 1.0, got {self.score}."
            )


class OcclusionDetector(ABC):
    """
    Interface for semantic occlusion detection.
    """

    @abstractmethod
    def assess(self, image: np.ndarray) -> OcclusionResult:
        """
        Assess an image for semantic occlusion.

        Args:
            image: Input image as a NumPy array.

        Returns:
            OcclusionResult containing the quality score and detection state.
        """
        raise NotImplementedError


class HeuristicOcclusionDetector(OcclusionDetector):
    """
    Classical, no-training heuristic occlusion detector.

    Kept as a dependency-free fallback / test double now that
    ModelOcclusionDetector is the real implementation. See that
    class's docstring for the project's actual occlusion strategy.

    Method: partial/full occlusion (a finger over the lens, a sticker,
    a solid overlay, a UI element) tends to create a large, contiguous,
    texturally FLAT region. This detector tiles the image into blocks,
    measures local variance per block, flags low-variance blocks as
    "flat", and finds the largest connected flat region.

    Known false-positive modes: genuinely uniform backgrounds (sky,
    studio backdrop, out-of-focus bokeh) will also score as "flat".
    """

    def __init__(
        self,
        block_size: int = 16,
        flatness_std_threshold: float = 5.0,
        min_occlusion_area_ratio: float = 0.15,
    ) -> None:
        self.block_size = block_size
        self.flatness_std_threshold = flatness_std_threshold
        self.min_occlusion_area_ratio = min_occlusion_area_ratio

    def _flat_block_mask(self, gray: np.ndarray) -> np.ndarray:
        block = self.block_size
        gray_f = gray.astype(np.float32)
        mean = cv2.blur(gray_f, (block, block))
        mean_sq = cv2.blur(gray_f * gray_f, (block, block))
        variance = np.clip(mean_sq - mean * mean, 0, None)
        std = np.sqrt(variance)
        return (std < self.flatness_std_threshold).astype(np.uint8) * 255

    def assess(self, image: np.ndarray) -> OcclusionResult:
        validate_image(image, min_dimension=MIN_DIMENSION)
        gray = to_grayscale(image)

        flat_mask = self._flat_block_mask(gray)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(flat_mask, connectivity=8)

        if num_labels <= 1:
            largest_area_ratio = 0.0
        else:
            region_areas = stats[1:num_labels, cv2.CC_STAT_AREA]
            largest_area_ratio = float(np.max(region_areas) / flat_mask.size)

        detected = largest_area_ratio >= self.min_occlusion_area_ratio
        score = float(np.clip(1.0 - largest_area_ratio, 0.0, 1.0))

        return OcclusionResult(score=score, detected=detected)


class ModelOcclusionDetector(OcclusionDetector):
    """
    Face-occlusion detector using a pretrained ConvNeXt-Small classifier
    (LamKser/face-occlusion-classification, MIT license; ONNX export via
    Jacky622/face_occlusion on HuggingFace).

    SCOPING NOTE (project decision): the classifier was trained ONLY on
    cropped face images — binary occluded/non-occluded face
    classification. It has no meaning applied to a generic scene with
    no face in it. So this detector:
      - If a face is found: crops it and runs the classifier, returning
        a genuine occlusion score/verdict (face_detected=True).
      - If no face is found: reports occlusion as "not applicable"
        (face_detected=False, detected=False, score=1.0) instead of
        misapplying a face-only model to arbitrary content. Every other
        detector (blur, glare, exposure, motion, resolution, framing)
        still runs normally on such images elsewhere in the pipeline —
        this detector simply opts out for that one image.

    FACE DETECTOR CHOICE — YuNet, not Haar cascade:
    An earlier version used OpenCV's bundled Haar cascade for the
    face-detection gate. In testing, Haar cascade failed to find a face
    at all on mask-occluded faces (it relies on matching eye/nose/mouth
    pattern structure, which a mask disrupts) — so occlusion was
    silently reported as "not applicable" on exactly the images that
    most needed a real verdict. YuNet (cv2.FaceDetectorYN, from
    opencv_zoo) is a small CNN detector trained on WIDER FACE, which
    includes partially-occluded faces, and is materially more robust
    to masks/hands/angled faces than a Haar cascade while still being a
    small bundled-style ONNX model with no new heavy dependency (uses
    the same cv2 already required elsewhere in this project).

    Known remaining limitation (separate issue, being addressed via
    fine-tuning on COFW): even once a face IS detected, the ConvNeXt
    classifier itself under-detects hand-over-face occlusion, because
    its training data (mask/sunglasses-style crawled images) doesn't
    represent that occlusion type well. This class only fixes the
    face-detection stage; the classifier's own generalization gap is a
    separate, tracked problem.
    """

    INPUT_SIZE = 224
    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        onnx_model_path: str | Path,
        face_detector_model_path: str | Path,
        occlusion_threshold: float = 0.5,
        face_score_threshold: float = 0.6,
        face_nms_threshold: float = 0.3,
        face_top_k: int = 5000,
    ) -> None:
        """
        Args:
            onnx_model_path: Path to face_occlusion.onnx (ConvNeXt-Small,
                opset 12). Input: [1,3,224,224] float32 RGB, scaled to
                [0,1] then ImageNet-normalized, NCHW. Output: [1,2]
                logits — softmax index 1 = occluded.
            face_detector_model_path: Path to a YuNet ONNX model, e.g.
                face_detection_yunet_2023mar.onnx from opencv_zoo
                (https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
            occlusion_threshold: Softmax probability (index 1) at or
                above which a face is called occluded. This is the
                upstream classifier's own 0.5 decision boundary — not
                yet recalibrated against this project's own data.
            face_score_threshold: Minimum YuNet confidence for a
                detection to count as a face. YuNet's own default demo
                value is 0.9; lowered here to 0.6 because occluded
                faces legitimately score lower confidence than clean
                ones, and rejecting them at the detection stage would
                reproduce the same silent-skip problem Haar cascade had.
            face_nms_threshold: IoU threshold for YuNet's internal
                non-max suppression.
            face_top_k: Max candidate boxes YuNet keeps before NMS.

        Raises:
            ImportError: If onnxruntime is not installed.
            FileNotFoundError: If either ONNX model can't be loaded.
        """
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ModelOcclusionDetector. "
                "Install it with `pip install onnxruntime`."
            )

        onnx_model_path = Path(onnx_model_path)
        if not onnx_model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

        face_detector_model_path = Path(face_detector_model_path)
        if not face_detector_model_path.is_file():
            raise FileNotFoundError(
                f"YuNet face detector model not found: {face_detector_model_path}"
            )

        self.occlusion_threshold = occlusion_threshold

        self._session = ort.InferenceSession(
            str(onnx_model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        # input_size is set per-image in _detect_largest_face (YuNet
        # needs it to match the actual image dimensions), so (0, 0)
        # here is just a placeholder.
        self._face_detector = cv2.FaceDetectorYN.create(
            str(face_detector_model_path),
            "",
            (0, 0),
            score_threshold=face_score_threshold,
            nms_threshold=face_nms_threshold,
            top_k=face_top_k,
        )

    def _detect_largest_face(self, image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        height, width = image_bgr.shape[:2]
        self._face_detector.setInputSize((width, height))
        _, faces = self._face_detector.detect(image_bgr)

        if faces is None or len(faces) == 0:
            return None

        # Each row: [x, y, w, h, <5 landmark points>, score]. Take the
        # largest box by area, same "largest = subject" stand-in used
        # elsewhere in this project pending a real subject detector.
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest[:4]

        # Clip to image bounds — YuNet can return boxes that slightly
        # overshoot the edges.
        x = int(max(0, x))
        y = int(max(0, y))
        w = int(min(w, width - x))
        h = int(min(h, height - y))

        if w <= 0 or h <= 0:
            return None

        return (x, y, w, h)

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            face_rgb, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR
        )
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self._IMAGENET_MEAN) / self._IMAGENET_STD
        chw = normalized.transpose(2, 0, 1)
        return np.expand_dims(chw, axis=0).astype(np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    def assess(self, image: np.ndarray) -> OcclusionResult:
        validate_image(image, min_dimension=MIN_DIMENSION)

        if image.ndim != 3:
            raise ValueError(
                "ModelOcclusionDetector requires a color (BGR/BGRA) image "
                "for face detection; grayscale input is not supported."
            )

        image_bgr = image[:, :, :3] if image.shape[2] == 4 else image

        bbox = self._detect_largest_face(image_bgr)
        if bbox is None:
            # No face in the image -> the occlusion model doesn't apply.
            # Reported as "not applicable", not as "clean" or "occluded".
            return OcclusionResult(score=1.0, detected=False, face_detected=False)

        x, y, w, h = bbox
        face_crop = image_bgr[y : y + h, x : x + w]

        input_tensor = self._preprocess(face_crop)
        logits = self._session.run(None, {self._input_name: input_tensor})[0][0]
        probabilities = self._softmax(logits)
        occluded_probability = float(probabilities[1])

        score = float(np.clip(1.0 - occluded_probability, 0.0, 1.0))
        detected = occluded_probability >= self.occlusion_threshold

        return OcclusionResult(score=score, detected=detected, face_detected=True)