# Engineering Decisions

Chronological record of the non-obvious calls made on this project, and why.

## Occlusion: pretrained ConvNeXt-Small, face-gated

**Decision:** Use `LamKser/face-occlusion-classification`'s ConvNeXt-Small
(ONNX export via HuggingFace `Jacky622/face_occlusion`) rather than training
from scratch.

**Why:** The model was trained specifically for face-occlusion binary
classification (Val accuracy 0.9887 / F1 0.9884 upstream) — reusable rather
than building an occlusion classifier from zero.

**Important scoping caveat:** the model was trained ONLY on cropped face
images. It has no meaning applied to a generic scene with no face. So
`ModelOcclusionDetector`:
- If a face is found → crops it, runs the classifier, returns a real verdict
- If no face is found → reports `face_detected: false`, `score: 1.0`,
  `detected: false` — "not applicable," not a forced "clean" verdict

## Occlusion: Haar cascade → YuNet

**Problem found:** Haar cascade (`haarcascade_frontalface_default.xml`)
failed to detect a face AT ALL on mask-occluded images — it relies on
matching eye/nose/mouth pattern structure, which a mask disrupts. Result:
occlusion was silently reported as "not applicable" on exactly the images
that most needed a real verdict.

**Decision:** Switched to `cv2.FaceDetectorYN` (YuNet), trained on WIDER
FACE (includes partially-occluded faces). `face_score_threshold` set to 0.6
(YuNet's own demo default is 0.9) — occluded faces legitimately score lower
confidence, and a high threshold would reproduce the same silent-skip
problem.

**Confirmed fix:** `mask.jpeg` went from `face_detected: false` (Haar) to
`face_detected: true, score: 0.0000069, detected: true` (YuNet) — correctly
detected AND correctly classified as occluded.

## Occlusion: fine-tuning on COFW, not a threshold hack

**Problem found:** the ConvNeXt-Small classifier confidently detected
mask/sunglasses-style occlusion but failed on real hand-over-face occlusion
(20.9%/10.7% confidence on two real test images, even worse on tight crops).
Root cause: the model's original training data was crawled and face-cropped
via `FaceMaskDetection` (a COVID-era mask detector) — skewed toward
mask/sunglasses occlusion, not hand occlusion. This is a training-data
domain gap, not a threshold-calibration problem.

**Rejected approach:** ensembling with the classical heuristic, or just
lowering the decision threshold — both would paper over the gap rather than
close it.

**Decision:** Fine-tune on COFW (Caltech Occluded Faces in the Wild —
real-world hand/hair/object occlusion, exactly the missing case).

**Fine-tuning design choices:**
- **Partial freezing** (`features[0:6]` frozen, only later stages +
  classifier trainable): COFW's usable training data (~500 genuinely
  occluded images) is much smaller than the original ~9,749-image training
  set — fine-tuning the whole network risked catastrophic forgetting of the
  mask/sunglasses cases it already handled well.
- **Low LR (1e-5)** — fine-tuning, not training from scratch.
- **Class-weighted loss** — train split is 998 clean / 347 occluded.
- **+30% expanded face crop** — a tight crop (just the bbox) was found
  during evaluation to make occlusion even harder to detect, likely because
  it crops out the edges (hand/hair boundary against skin) that signal
  occlusion.

**Result:** best val_acc 0.7870 (epoch 14/15). Confirmed no regression on
`mask.jpeg` after fine-tuning (0.000007 → 0.013 occluded-probability,
still confidently correct).

## Framing: U²-NetP for subject localization, not a framing verdict

**Decision:** Replace the Canny-edge/largest-contour heuristic's subject
detection with U²-NetP (salient object detection), keeping the geometry/
scoring logic separate.

**Important correction made during design review:** U²-Net provides
*subject localization*, not framing quality by itself. A salient region
isn't always the intended subject — landscapes, cluttered scenes, and group
shots can produce a salient region a human wouldn't call "the subject."
Architecture is therefore strictly:

```
U²-NetP saliency mask -> geometric measurements -> FramingResult (own scoring rule)
```

not "U²-Net directly outputs a framing score."

**Also corrected:** an off-center subject is not automatically bad framing
(compositional choice) — `subject_position_score` contributes only a minor
weight (`FramingThresholds.position_weight = 0.2`) to the final score,
not a hard "center = good" rule.

**Thresholds centralized**, not scattered as inline magic numbers — see
`FramingThresholds` in `app/quality/semantic/framing.py`. None are
calibrated against labeled data yet; all are documented starting guesses
(e.g. `edge_margin_threshold = 0.02` — an *initial* guess for "touching the
edge," explicitly flagged for recalibration against real cropped/uncropped
examples, not treated as ground truth).

**`HeuristicFramingDetector` kept, not deleted** — same policy as occlusion:
the model detector is an alternative/primary implementation, not a hard
replacement, until real-data evaluation justifies retiring the heuristic.

## Suitability model: synthetic-injection labels, not detector-derived labels

**Problem avoided:** if ground-truth labels were derived from the
detectors' own outputs (e.g. "detectors flagged it bad → label it Not
Suitable"), the model would just be learning to reproduce the detectors —
circular, no new signal.

**Decision:** Inject controlled, severity-graded synthetic defects into
clean images. The **injection severity** (known exactly, since we chose it)
becomes the label; the **detectors' outputs** on the resulting degraded
image become the features. These are independent by construction.

**Label rule:** Not Suitable if any defect is "high" severity, OR 2+
defects are "medium" severity.

**Darkness/Overexposure made mutually exclusive** (`sample_exposure_severity`
picks at most one) — both are opposite ends of the same brightness axis; an
image can't be validly both injected-dark and injected-bright at once.

**Resolution injected via real resize, not blur** — KonIQ-10k is a fixed
512×384, so a blur-based "fake low-res" trick wouldn't change the actual
pixel dimensions the Resolution detector measures. `apply_resolution()`
instead resizes to real target dimensions (1024×768 / 800×600 / 480×360 /
240×180), straddling the API's 640×480 default threshold, preserving
KonIQ's native 4:3 aspect ratio.

**Occlusion not synthetically injected** — COFW's real
`occluded_landmark_ratio` is used instead (see `docs/dataset.md`); a
synthetic overlay would misrepresent what real occlusion looks like to the
detector.

**Model choice:** Random Forest over Logistic Regression — confirmed by
comparison (79% vs. 65% accuracy on the first 7-feature run). Expected:
the label rule ("high OR 2+ medium") is a non-linear threshold/count-based
rule that a linear model struggles to represent; tree-based models capture
it naturally.

**Framing feature added later, not at first**: the Suitability model was
trained and shipped on the original 7 features before Framing was wired
into the API, specifically to avoid blocking suitability on framing's
completion. Once framing was implemented and validated, the training set
was regenerated to include it and the model retrained — a deliberate
staged rollout rather than a big-bang integration.

## Production dependency split (Docker)

**Decision:** `requirements.txt` (Docker image, runtime-only) vs.
`requirements-train.txt` (dev venv only — `torch`/`torchvision`,
`scipy`/`h5py` for COFW extraction, training-time `pandas`, etc.).

**Why:** training (PyTorch, fine-tuning, dataset generation) is a separate,
offline concern from serving (ONNX inference + Pydantic + FastAPI). `torch`
never needs to be in the production image — its only job is producing
`.onnx`/`.joblib` artifacts that the API then loads.

**Amendment:** once the Suitability model moved from a training artifact to
something the API actually loads and runs at request time,
`scikit-learn`/`joblib`/`pandas` moved INTO `requirements.txt` — unpickling
and running a fitted sklearn `Pipeline` requires `scikit-learn` installed,
not just `joblib`.

**Models baked into the Docker image** (`COPY models/ ./models/`), not
runtime-volume-mounted — chosen for a self-contained, portable image over a
smaller image with an external dependency. Confirmed working: full
`docker build` + `docker run` cycle, clean startup, correct `/assess`
results matching the non-Docker run.