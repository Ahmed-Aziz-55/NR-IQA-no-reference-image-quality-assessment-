"""
Trains the final Suitability classifier on the CSV produced by
generate_training_dataset.py.

Compares:
    1. Logistic Regression
    2. Random Forest

The model uses only detector-output features.

Important:
    *_applicable features distinguish "detector not applicable"
    from an actually good detector score.

Usage:
    python train_suitability_model.py \
        --training-csv data/suitability_training.csv \
        --output-dir models/
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Features used by the suitability model
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Blur
    "blur_laplacian_variance",

    # Exposure
    "exposure_mean_brightness",
    "exposure_dark_pixel_ratio",
    "exposure_bright_pixel_ratio",
    "exposure_saturated_pixel_ratio",

    # Glare
    "glare_area_ratio",
    "glare_region_count",
    "glare_largest_region_ratio",

    # Motion
    "motion_gradient_variance_x",
    "motion_gradient_variance_y",
    "motion_blur_ratio",
    "motion_overall_sharpness",
    "motion_is_likely_motion_blur",

    # Resolution
    "resolution_width",
    "resolution_height",
    "resolution_total_pixels",
    "resolution_aspect_ratio",
    "resolution_is_below_minimum",

    # Occlusion
    "occlusion_score",
    "occlusion_detected",
    "occlusion_face_detected",
    "occlusion_applicable",

    # Framing
    "framing_score",
    "framing_detected",
    "framing_applicable",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Suitability classifier."
    )

    parser.add_argument(
        "--training-csv",
        type=Path,
        required=True,
        help="Path to suitability training CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the trained model will be saved.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for testing.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------

    if not args.training_csv.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {args.training_csv}"
        )

    df = pd.read_csv(args.training_csv)

    print(f"Loaded training dataset: {len(df)} rows")

    # -----------------------------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------------------------

    required_columns = FEATURE_COLUMNS + ["suitable"]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Training CSV is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
            + "\n\n"
            "Make sure generate_training_dataset.py produces "
            "the updated feature columns."
        )

    # -----------------------------------------------------------------------
    # Validate target
    # -----------------------------------------------------------------------

    if df["suitable"].isna().any():
        raise ValueError(
            "Training CSV contains missing values in 'suitable'."
        )

    # -----------------------------------------------------------------------
    # Validate feature values
    # -----------------------------------------------------------------------

    X = df[FEATURE_COLUMNS].copy()
    y = df["suitable"].copy()

    # Convert boolean columns to integers.
    boolean_columns = [
        "motion_is_likely_motion_blur",
        "resolution_is_below_minimum",
        "occlusion_detected",
        "occlusion_face_detected",
        "occlusion_applicable",
        "framing_detected",
        "framing_applicable",
    ]

    for column in boolean_columns:
        X[column] = X[column].astype(int)

    # Ensure all features are numeric.
    for column in FEATURE_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="raise",
        )

    # Check for NaN / infinity.
    if X.isna().any().any():
        bad_columns = X.columns[X.isna().any()].tolist()

        raise ValueError(
            f"Training data contains NaN values in: {bad_columns}"
        )

    # -----------------------------------------------------------------------
    # Print applicability statistics
    # -----------------------------------------------------------------------

    print("\n=== Detector Applicability ===")

    print(
        "Occlusion applicable:",
        int(X["occlusion_applicable"].sum()),
        "/",
        len(X),
    )

    print(
        "Framing applicable:",
        int(X["framing_applicable"].sum()),
        "/",
        len(X),
    )

    # -----------------------------------------------------------------------
    # Train/test split
    # -----------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    print("\n=== Dataset Split ===")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")

    # -----------------------------------------------------------------------
    # Logistic Regression
    # -----------------------------------------------------------------------

    logreg_pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )

    logreg_pipeline.fit(
        X_train,
        y_train,
    )

    logreg_predictions = logreg_pipeline.predict(X_test)

    logreg_f1 = f1_score(
        y_test,
        logreg_predictions,
    )

    print("\n=== Logistic Regression ===")

    print(
        classification_report(
            y_test,
            logreg_predictions,
            target_names=[
                "not_suitable",
                "suitable",
            ],
            zero_division=0,
        )
    )

    # -----------------------------------------------------------------------
    # Random Forest
    # -----------------------------------------------------------------------

    rf_pipeline = Pipeline(
        [
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=args.seed,
                    n_jobs=-1,
                ),
            )
        ]
    )

    rf_pipeline.fit(
        X_train,
        y_train,
    )

    rf_predictions = rf_pipeline.predict(X_test)

    rf_f1 = f1_score(
        y_test,
        rf_predictions,
    )

    print("\n=== Random Forest ===")

    print(
        classification_report(
            y_test,
            rf_predictions,
            target_names=[
                "not_suitable",
                "suitable",
            ],
            zero_division=0,
        )
    )

    # -----------------------------------------------------------------------
    # Select best model
    # -----------------------------------------------------------------------

    if logreg_f1 >= rf_f1:
        best_name = "logistic_regression"
        best_pipeline = logreg_pipeline
        best_f1 = logreg_f1
    else:
        best_name = "random_forest"
        best_pipeline = rf_pipeline
        best_f1 = rf_f1

    print(
        f"\nBest model: {best_name} "
        f"(F1={best_f1:.4f})"
    )

    # -----------------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        args.output_dir
        / "suitability_model.joblib"
    )

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "feature_columns": FEATURE_COLUMNS,
        },
        model_path,
    )

    # -----------------------------------------------------------------------
    # Save metadata
    # -----------------------------------------------------------------------

    metadata = {
        "model_type": best_name,
        "f1_score": best_f1,
        "logistic_regression_f1": logreg_f1,
        "random_forest_f1": rf_f1,
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": args.test_size,
        "random_seed": args.seed,
        "occlusion_applicable_rows": int(
            X["occlusion_applicable"].sum()
        ),
        "framing_applicable_rows": int(
            X["framing_applicable"].sum()
        ),
    }

    metadata_path = (
        args.output_dir
        / "suitability_model_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()