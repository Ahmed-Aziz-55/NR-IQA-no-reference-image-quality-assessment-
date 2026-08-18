# System Architecture

(To be filled in as components are added.)
# Architecture

## Module layout

```
app/
├── api/
│   ├── main.py         FastAPI app, model loading (once, at startup), /assess route
│   ├── assess.py        fan-out: runs every detector, then the Suitability model
│   └── schemas.py        Pydantic response models
├── io/
│   └── image_loader.py   disk -> validated BGR numpy array
└── quality/
    ├── classical/         no-training detectors: cheap, deterministic, fast
    │   ├── _image_utils.py   shared validation/grayscale helpers
    │   ├── blur.py
    │   ├── exposure.py       darkness + overexposure
    │   ├── glare.py
    │   ├── motion.py
    │   └── resolution.py
    └── semantic/           model-based detectors: need a subject/face concept
        ├── occlusion.py     ModelOcclusionDetector (+ legacy HeuristicOcclusionDetector)
        └── framing.py       ModelFramingDetector (+ legacy HeuristicFramingDetector)
```

## Why classical vs. semantic

`classical/` detectors work on pixel statistics alone (variance, HSV
thresholds, gradients) — no notion of "subject," no model weights, sub-
millisecond, and fully deterministic. They cover Blur, Darkness,
Overexposure, Glare, Motion, and Resolution.

`semantic/` detectors need a concept of "what's in the image" — occlusion
only means something relative to a face; framing only means something
relative to a subject. Both are model-based (ONNX inference), and both
follow the same internal shape:

```
model inference -> raw prediction -> geometry/scoring (pure function) -> Result dataclass
```

Keeping the scoring logic in a pure function (`compute_framing_result()` in
framing.py; the equivalent logic is inline in occlusion.py's `assess()`)
means it's unit-testable with synthetic inputs, with no ONNX model file
required at test time — see `tests/unit/quality/`.

## Detector interface pattern

Every semantic detector follows the same ABC + dataclass shape:

```python
class XDetector(ABC):
    @abstractmethod
    def assess(self, image: np.ndarray) -> XResult: ...

class HeuristicXDetector(XDetector):   # classical fallback, kept, not deleted
    ...

class ModelXDetector(XDetector):        # primary implementation
    ...
```

Both `HeuristicOcclusionDetector`/`HeuristicFramingDetector` remain in the
codebase as fallbacks/test doubles — the model-based detectors were
integrated as the primary implementation, not a hard replacement, until
evaluation on real data justifies retiring the heuristics (see
`docs/decisions.md`).

## Request flow (`POST /assess`)

1. `main.py` — upload validated (extension, size), saved to a temp file
2. `image_loader.load_image()` — decoded to a BGR numpy array, validated
3. `assess.py:assess_image()` — runs every detector in sequence:
   - blur, exposure, glare, motion, resolution (classical, cheap)
   - occlusion (YuNet face detection -> ConvNeXt-Small classifier, only if a face is found)
   - framing (U²-NetP saliency -> geometric scoring, only if a subject is found)
   - suitability (feature vector built from the above -> trained RandomForest)
4. Assembled into `AssessmentResponse`, returned as JSON

Every detector is a pure function of the image except the model-based ones,
which take a pre-loaded detector instance — loaded ONCE at API startup in
`main.py`, not per-request. This is why `main.py` has module-level
`_occlusion_detector`, `_framing_detector`, `_suitability_model` globals:
ONNX session creation and model deserialization are too expensive to repeat
per-request.

## Why "raw metrics, one opinionated field"

`schemas.py`'s `AssessmentResponse` is deliberately flat: every detector
field is a raw, un-thresholded metric (`laplacian_variance: 3.6`, not
`is_blurry: true`). This lets callers apply their own downstream-model-
specific cutoffs (see `docs/features/resolution.md` for why a fixed
threshold is actively wrong for some use cases).

`suitability` is the one deliberate exception — a single combined verdict
from the trained model, for callers who just want a yes/no answer without
reasoning about 8 separate metrics themselves.

## Production dependency split

`requirements.txt` (Docker image) vs. `requirements-train.txt` (dev venv
only) — see `docs/decisions.md` for the reasoning. Runtime needs
`scikit-learn`/`joblib`/`pandas` too now (to unpickle and run the
Suitability model), not just `onnxruntime` — this was a deliberate
addition once the Suitability model moved from "training artifact" to
"served in production."