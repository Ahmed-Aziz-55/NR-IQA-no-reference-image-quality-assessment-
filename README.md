# IQA — Image Quality Assessment for Downstream CV Suitability

Status: implementation in progress, feature-by-feature, against a locked research plan.

See `docs/research.md` for the full locked plan and `docs/decisions.md` for
engineering decisions made where research left gaps.

## Features (8)
1. Blur
2. Darkness / Underexposure
3. Overexposure
4. Glare
5. Motion Artifacts
6. Occlusion
7. Poor Framing
8. Low Resolution

Final suitability score is calibrated against real downstream CV task
performance — not a hand-picked formula. See `docs/evaluation.md`.
