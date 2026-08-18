"""
Builds the Suitability model's training CSV.

Sources:

1. KonIQ-10k
   Generic images used primarily for the classical quality defects.

2. COFW
   Face images with real occlusion ground truth.

The production FeatureExtractor is reused so the suitability model
is trained on the same detector signals used during inference.

Usage:

    python scripts/generate_training_dataset.py \
        --koniq-dir koniq10k_512x384 \
        --cofw-labels-csv data/cofw_extracted/labels.csv \
        --occlusion-onnx models/face_occlusion_finetuned.onnx \
        --face-detector-onnx models/face_detection_yunet_2023mar.onnx \
        --framing-onnx models/u2netp.onnx \
        --output-csv data/suitability_training.csv \
        --koniq-base-count 1500 \
        --koniq-variants-per-image 2 \
        --cofw-count 1000 \
        --seed 42
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

# Make project root importable.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from app.quality.semantic.occlusion import ModelOcclusionDetector
from app.quality.semantic.framing import ModelFramingDetector

from feature_extractor import FeatureExtractor

from synthetic_augmentation import (
    InjectedSeverities,
    SyntheticAugmentationEngine,
    occlusion_severity_from_ratio,
)


def _list_images(directory: Path) -> list[Path]:
    """
    Return supported image files from a directory.
    """

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }

    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def _process_and_append(
    image: np.ndarray,
    severities: InjectedSeverities,
    engine: SyntheticAugmentationEngine,
    extractor: FeatureExtractor,
    source: str,
    rows: list[dict],
) -> None:
    """
    Apply synthetic degradation, extract production features,
    and append one training row.
    """

    # ---------------------------------------------------------------
    # Apply synthetic defects
    # ---------------------------------------------------------------

    augmented = engine.apply(
        image,
        severities,
    )

    # ---------------------------------------------------------------
    # Extract production detector features
    # ---------------------------------------------------------------

    features = extractor.extract(
        augmented
    )

    # Start row with detector outputs.
    row = dict(features)

    # ---------------------------------------------------------------
    # Ground-truth / debug metadata
    # ---------------------------------------------------------------

    for defect, severity in severities.as_dict().items():
        row[f"severity_{defect}"] = severity

    # Final suitability label.
    row["suitable"] = int(
        severities.is_suitable()
    )

    # Dataset source.
    row["source"] = source

    rows.append(row)


def generate_from_koniq(
    koniq_dir: Path,
    base_count: int,
    variants_per_image: int,
    engine: SyntheticAugmentationEngine,
    extractor: FeatureExtractor,
    rng: np.random.Generator,
    rows: list[dict],
) -> None:
    """
    Generate training examples from KonIQ-10k.
    """

    all_images = _list_images(
        koniq_dir
    )

    if not all_images:
        raise FileNotFoundError(
            f"No images found in {koniq_dir}"
        )

    chosen = rng.choice(
        all_images,
        size=min(
            base_count,
            len(all_images),
        ),
        replace=False,
    )

    total = len(chosen)

    for i, image_path in enumerate(chosen):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"Warning: could not read {image_path}"
            )
            continue

        for _ in range(
            variants_per_image
        ):
            # KonIQ is primarily used for generic image-quality
            # degradation. No explicit occlusion ground truth is
            # injected here.
            severities = (
                engine.sample_severities(
                    occlusion_severity="none"
                )
            )

            _process_and_append(
                image=image,
                severities=severities,
                engine=engine,
                extractor=extractor,
                source="koniq",
                rows=rows,
            )

        if (i + 1) % 200 == 0:
            print(
                f"  KonIQ: processed "
                f"{i + 1}/{total} base images"
            )


def generate_from_cofw(
    cofw_labels_csv: Path,
    count: int,
    engine: SyntheticAugmentationEngine,
    extractor: FeatureExtractor,
    rng: np.random.Generator,
    rows: list[dict],
) -> None:
    """
    Generate training examples from COFW.

    COFW provides real face-occlusion ground truth through
    occluded_landmark_ratio.
    """

    labels_df = pd.read_csv(
        cofw_labels_csv
    )

    if labels_df.empty:
        raise ValueError(
            f"COFW labels CSV is empty: {cofw_labels_csv}"
        )

    required_columns = {
        "image_path",
        "occluded_landmark_ratio",
    }

    missing = (
        required_columns
        - set(labels_df.columns)
    )

    if missing:
        raise ValueError(
            "COFW labels CSV is missing columns: "
            f"{sorted(missing)}"
        )

    sample_size = min(
        count,
        len(labels_df),
    )

    sampled = labels_df.sample(
        n=sample_size,
        random_state=int(
            rng.integers(
                0,
                2**31,
            )
        ),
    ).reset_index(
        drop=True
    )

    for i, record in sampled.iterrows():

        image_path = Path(
            record["image_path"]
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"Warning: could not read "
                f"{image_path}"
            )
            continue

        # Convert real COFW occlusion annotation
        # into the severity representation used by
        # the augmentation/training pipeline.
        occlusion_severity = (
            occlusion_severity_from_ratio(
                record[
                    "occluded_landmark_ratio"
                ]
            )
        )

        severities = (
            engine.sample_severities(
                occlusion_severity=occlusion_severity
            )
        )

        _process_and_append(
            image=image,
            severities=severities,
            engine=engine,
            extractor=extractor,
            source="cofw",
            rows=rows,
        )

        if (i + 1) % 200 == 0:
            print(
                f"  COFW: processed "
                f"{i + 1}/{sample_size} images"
            )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate the Suitability model "
            "training CSV."
        )
    )

    # ---------------------------------------------------------------
    # Dataset paths
    # ---------------------------------------------------------------

    parser.add_argument(
        "--koniq-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cofw-labels-csv",
        type=Path,
        required=True,
    )

    # ---------------------------------------------------------------
    # ONNX models
    # ---------------------------------------------------------------

    parser.add_argument(
        "--occlusion-onnx",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--face-detector-onnx",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--framing-onnx",
        type=Path,
        required=True,
        help="Path to U2NetP ONNX model.",
    )

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )

    # ---------------------------------------------------------------
    # Dataset sizes
    # ---------------------------------------------------------------

    parser.add_argument(
        "--koniq-base-count",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--koniq-variants-per-image",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--cofw-count",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Validate model paths
    # ---------------------------------------------------------------

    required_paths = {
        "KonIQ directory": args.koniq_dir,
        "COFW labels CSV": args.cofw_labels_csv,
        "Occlusion ONNX": args.occlusion_onnx,
        "Face detector ONNX": args.face_detector_onnx,
        "Framing ONNX": args.framing_onnx,
    }

    for name, path in required_paths.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found: {path}"
            )

    # ---------------------------------------------------------------
    # Random generator
    # ---------------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    # ---------------------------------------------------------------
    # Synthetic augmentation engine
    # ---------------------------------------------------------------

    engine = SyntheticAugmentationEngine(
        rng=rng
    )

    # ---------------------------------------------------------------
    # Production occlusion detector
    # ---------------------------------------------------------------

    occlusion_detector = (
        ModelOcclusionDetector(
            onnx_model_path=args.occlusion_onnx,
            face_detector_model_path=(
                args.face_detector_onnx
            ),
        )
    )

    # ---------------------------------------------------------------
    # Production framing detector
    # ---------------------------------------------------------------

    framing_detector = (
        ModelFramingDetector(
            onnx_model_path=args.framing_onnx
        )
    )

    # ---------------------------------------------------------------
    # Feature extractor
    # ---------------------------------------------------------------

    extractor = FeatureExtractor(
        occlusion_detector=occlusion_detector,
        framing_detector=framing_detector,
    )

    # ---------------------------------------------------------------
    # Generate rows
    # ---------------------------------------------------------------

    rows: list[dict] = []

    print(
        "Generating from KonIQ-10k..."
    )

    generate_from_koniq(
        koniq_dir=args.koniq_dir,
        base_count=args.koniq_base_count,
        variants_per_image=(
            args.koniq_variants_per_image
        ),
        engine=engine,
        extractor=extractor,
        rng=rng,
        rows=rows,
    )

    print(
        "\nGenerating from COFW..."
    )

    generate_from_cofw(
        cofw_labels_csv=args.cofw_labels_csv,
        count=args.cofw_count,
        engine=engine,
        extractor=extractor,
        rng=rng,
        rows=rows,
    )

    # ---------------------------------------------------------------
    # Create DataFrame
    # ---------------------------------------------------------------

    if not rows:
        raise RuntimeError(
            "No training rows were generated."
        )

    df = pd.DataFrame(
        rows
    )

    # ---------------------------------------------------------------
    # Validate new features
    # ---------------------------------------------------------------

    required_feature_columns = [
        "occlusion_score",
        "occlusion_detected",
        "occlusion_face_detected",
        "occlusion_applicable",
        "framing_score",
        "framing_detected",
        "framing_applicable",
    ]

    missing = [
        column
        for column in required_feature_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Generated dataset is missing required "
            f"feature columns: {missing}"
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        args.output_csv,
        index=False,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    suitable_count = int(
        df["suitable"].sum()
    )

    not_suitable_count = int(
        len(df) - suitable_count
    )

    print(
        f"\nWrote {len(df)} rows to "
        f"{args.output_csv}"
    )

    print(
        f"Suitable: {suitable_count} / "
        f"Not suitable: {not_suitable_count}"
    )

    print(
        "\n=== Applicability ==="
    )

    print(
        "Occlusion applicable:",
        int(
            df["occlusion_applicable"].sum()
        ),
        "/",
        len(df),
    )

    print(
        "Framing applicable:",
        int(
            df["framing_applicable"].sum()
        ),
        "/",
        len(df),
    )


if __name__ == "__main__":
    main()