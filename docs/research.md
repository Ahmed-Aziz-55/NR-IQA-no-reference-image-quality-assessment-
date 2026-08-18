# Research

Background research behind the two model-based detectors (occlusion,
framing) — what was considered, what was chosen, and why. See
`docs/decisions.md` for the resulting engineering decisions; this doc is
the "why" behind those.

## Occlusion

### Requirement
Detect whether a face in an image is occluded (hand, mask, sunglasses,
hair, object) — one of 8 quality-suitability signals.

### Candidates considered

**Train from scratch:** rejected — a pretrained, purpose-built occlusion
classifier already existed and was directly reusable.

**`LamKser/face-occlusion-classification` (ConvNeXt-Small):** chosen.
49.4M params, 0.9887 val accuracy / 0.9884 F1 on its own benchmark — the
strongest model in that repo's comparison. ONNX export available via
HuggingFace `Jacky622/face_occlusion`, avoiding a full PyTorch runtime
dependency in production (input: 224×224 RGB, ImageNet-normalized, NCHW;
output: 2-class logits).

### Face-detection stage: candidates considered

**Haar cascade** (`cv2.CascadeClassifier`, bundled with OpenCV): initially
used, no new dependency. **Failed** — relies on matching eye/nose/mouth
pattern structure, which occlusion (by definition) disrupts. Confirmed
failure case: 0% detection on mask-occluded test image.

**YuNet** (`cv2.FaceDetectorYN`, from `opencv_zoo`): chosen. Small CNN
detector trained on WIDER FACE, which includes partially-occluded faces —
materially more robust to masks/hands/angled faces, while still a small
ONNX model with no new heavy dependency (uses the same `cv2` already
required elsewhere in this project).

### Generalization gap: root cause investigation

**Symptom:** the ConvNeXt-Small classifier scored real hand-over-face
occlusion at 20.9%/10.7% occluded-probability — confidently WRONG (should
be high, near 100%). A 30%-expanded crop improved but did not fix this.

**Investigated and rejected as the cause:** the code's preprocessing,
threshold, or `detected` logic — verified correct.

**Root cause identified:** training-data domain gap. The original
9,749-image dataset was crawled and face-cropped via `FaceMaskDetection`
(a COVID-era face-mask detector) — this skews the training distribution
toward mask/sunglasses-style occlusion, underrepresenting generic
hand/object occlusion.

**Options considered for fixing it:**
1. Ensemble with the classical flat-region-variance heuristic — rejected,
   papers over the gap rather than closing it.
2. Try a different crop convention matching the original training pipeline
   — plausible, not pursued (fine-tuning was more direct).
3. Try other pretrained checkpoints from the same repo (VGG16/19,
   DenseNet169, ResNet) — not pursued, likely share the same training-data
   domain gap since they're trained on the same dataset.
4. **Fine-tune on a dataset with real hand/object occlusion — chosen.**
5. Drop the model, use only the classical heuristic — rejected, the
   heuristic has its own known false-positive modes (uniform backgrounds).

### Fine-tuning dataset: COFW

**Caltech Occluded Faces in the Wild (COFW)** — chosen specifically
because it's built around realistic partial occlusions (hands, hair,
sunglasses, objects), which is exactly the gap identified above. Widely
used occlusion-robustness benchmark. Per-landmark occlusion ground truth
(29 landmarks/face, occlusion bit per landmark) gives real, not synthetic,
occlusion labels.

## Framing

### Requirement
Assess whether the subject of an image is well-framed (not cropped, not
too small/large in the frame, reasonably positioned) — generalizing
beyond faces, since the project's suitability goal covers arbitrary
downstream-CV images, not just portraits.

### Candidates considered

**Class-restricted object detectors (YOLO, COCO-trained models):**
rejected as the primary approach — limited to ~80 fixed classes, fails on
subjects outside that vocabulary (most product photography, most animals/
objects outside COCO's list, abstract or unusual subjects).

**Face detection alone (reusing YuNet):** rejected as the sole approach —
generalizes only to portraits, not the project's broader "any downstream-
CV image" scope.

**Salient Object Detection (U²-Net / U²-NetP):** chosen. Class-agnostic —
finds the most visually salient region regardless of what it is (person,
product, animal, vehicle). Published: Qin et al., "U²-Net: Going Deeper
with Nested U-Structure for Salient Object Detection," *Pattern
Recognition*, 2020. `u2netp` (the lightweight variant, ~4.7MB) chosen over
the full `u2net` (~176MB) for CPU inference speed; industry precedent for
this choice includes background-removal tools (`rembg`) built on the same
model family.

### Correction made during design review

**Initial overclaim (corrected):** framing U²-Net as "the best/perfect
solution" and implying "salient object = the subject that should determine
framing." This is often true but not always — landscapes, cluttered
scenes, and group shots can produce a salient region that isn't the
semantic "subject" a human would name.

**Resulting architectural correction:** U²-Net's job is scoped strictly to
*subject localization* (a saliency mask → bounding box). All actual
framing-quality scoring (edge-touching, size ratio, position) lives in a
separate, pure scoring function that operates on that localization — never
"U²-Net directly outputs a framing score." See `docs/decisions.md` for the
resulting `FramingThresholds` design and the specific corrections to the
edge-touching and centering logic (margin-based rather than binary;
position as a soft signal, not "center = good").

### Alternatives noted but not adopted
`rembg`'s newer supported models (ISNet, BiRefNet variants) exist and may
outperform U²-NetP on some inputs — not evaluated here; U²-NetP was
treated as "a sensible first implementation," not a final/optimal choice.
Recommended follow-up (not yet done): run both U²-NetP and a newer
alternative on a small held-out test set and compare mask quality before
treating U²-NetP as settled.