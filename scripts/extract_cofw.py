"""
Extract COFW dataset (.mat) into flat image files + a labels CSV, for
two downstream uses in this project:
  1. Ground truth for ConvNeXt occlusion classifier fine-tuning (an
     image-level occluded/clean label derived from per-landmark bits).
  2. A source of real face-containing images for the Suitability
     model's synthetic-augmentation dataset — occlusion is face-gated
     (see app/quality/semantic/occlusion.py), so the generic KonIQ-10k
     set alone won't exercise that feature; COFW fills that gap.

COFW format (Burgos-Artizzu et al. 2013, color release):
  IsTr / IsT         : cell array of training/test images
  bboxesTr / bboxesT : [N, 4] face bbox (left, top, width, height)
  phisTr / phisT     : [N, 87] landmark ground truth —
                        cols 0:29  = x coordinates
                        cols 29:58 = y coordinates
                        cols 58:87 = occlusion bit (0=visible, 1=occluded)

NOTE: in the training split, the first 845 images come from LFPW
(unoccluded faces) and only the remaining ~500 are genuine COFW
(occluded) — this skews class balance toward "clean" in the training
split specifically; keep that in mind when using is_occluded downstream.

The color .mat files are often large enough to be saved in MATLAB
v7.3 (HDF5) format, which scipy.io.loadmat cannot read — this script
tries scipy first and falls back to h5py automatically.

Usage:
    python extract_cofw.py \\
        --train-mat /path/to/cofw_train_color.mat \\
        --test-mat /path/to/cofw_test_color.mat \\
        --output-dir data/cofw_extracted
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

NUM_LANDMARKS = 29


def _load_mat(path: Path):
    """
    Load a COFW .mat file, handling both legacy (<7.3) and HDF5-based
    (>=7.3) MATLAB formats.

    Returns:
        A tuple (backend, mat_object) where backend is "scipy" or
        "h5py", identifying how to read mat_object's contents.
    """
    try:
        from scipy.io import loadmat

        return ("scipy", loadmat(str(path)))
    except NotImplementedError:
        # scipy raises this for v7.3 (HDF5) files specifically.
        pass
    except ValueError:
        pass

    import h5py

    return ("h5py", h5py.File(str(path), "r"))


def _iter_records(backend: str, mat, img_key: str, bbox_key: str, phis_key: str):
    """
    Yield (image_bgr, bbox, occlusion_ratio) for every record in a
    loaded COFW .mat file, regardless of scipy/h5py backend.
    """
    if backend == "scipy":
        images = mat[img_key].squeeze()
        bboxes = mat[bbox_key]
        phis = mat[phis_key]

        for i in range(len(images)):
            image = images[i]
            if image.ndim == 2:
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

            bbox = tuple(int(v) for v in bboxes[i])
            occlusion_bits = phis[i, 2 * NUM_LANDMARKS : 3 * NUM_LANDMARKS]
            occlusion_ratio = float(np.mean(occlusion_bits))

            yield image, bbox, occlusion_ratio

    elif backend == "h5py":
        image_refs = mat[img_key][:]
        bboxes = np.transpose(mat[bbox_key][:])
        phis = np.transpose(mat[phis_key][:])
        total = image_refs.shape[1] if image_refs.ndim == 2 else image_refs.shape[0]

        for i in range(total):
            ref = image_refs[0][i] if image_refs.ndim == 2 else image_refs[i]
            image = np.transpose(mat[ref][:])
            if image.ndim == 2:
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

            bbox = tuple(int(v) for v in bboxes[i])
            occlusion_bits = phis[i, 2 * NUM_LANDMARKS : 3 * NUM_LANDMARKS]
            occlusion_ratio = float(np.mean(occlusion_bits))

            yield image, bbox, occlusion_ratio

    else:  # pragma: no cover
        raise ValueError(f"Unknown backend: {backend}")


def extract_split(
    mat_path: Path,
    img_key: str,
    bbox_key: str,
    phis_key: str,
    split_name: str,
    output_dir: Path,
    csv_writer: csv.DictWriter,
    occluded_threshold: float,
) -> int:
    backend, mat = _load_mat(mat_path)
    images_dir = output_dir / "images" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for image, bbox, occlusion_ratio in _iter_records(
        backend, mat, img_key, bbox_key, phis_key
    ):
        filename = f"{split_name}_{count:04d}.jpg"
        image_path = images_dir / filename
        cv2.imwrite(str(image_path), image)

        csv_writer.writerow(
            {
                "image_path": str(image_path),
                "split": split_name,
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_w": bbox[2],
                "bbox_h": bbox[3],
                "occluded_landmark_ratio": round(occlusion_ratio, 4),
                "is_occluded": int(occlusion_ratio >= occluded_threshold),
            }
        )
        count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract COFW .mat files into images + a labels CSV."
    )
    parser.add_argument(
        "--train-mat", type=Path, required=True, help="Path to cofw_train_color.mat"
    )
    parser.add_argument(
        "--test-mat", type=Path, required=True, help="Path to cofw_test_color.mat"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write images/ and labels.csv into",
    )
    parser.add_argument(
        "--occluded-threshold",
        type=float,
        default=0.15,
        help=(
            "Fraction of occluded landmarks (0.0-1.0) at or above which "
            "an image is labeled is_occluded=1. Default 0.15 (~4+ of 29 "
            "landmarks) is a placeholder, not calibrated — inspect the "
            "resulting label distribution and adjust if it looks off."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_csv_path = args.output_dir / "labels.csv"

    fieldnames = [
        "image_path",
        "split",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "occluded_landmark_ratio",
        "is_occluded",
    ]

    with open(labels_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        train_count = extract_split(
            args.train_mat,
            "IsTr",
            "bboxesTr",
            "phisTr",
            "train",
            args.output_dir,
            writer,
            args.occluded_threshold,
        )
        test_count = extract_split(
            args.test_mat,
            "IsT",
            "bboxesT",
            "phisT",
            "test",
            args.output_dir,
            writer,
            args.occluded_threshold,
        )

    print(f"Extracted {train_count} train + {test_count} test images.")
    print(f"Labels written to {labels_csv_path}")


if __name__ == "__main__":
    main()