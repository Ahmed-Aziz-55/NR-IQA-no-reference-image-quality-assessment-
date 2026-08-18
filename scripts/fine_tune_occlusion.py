"""
Fine-tune the existing ConvNeXt-Small face-occlusion classifier
(best_convnext_small.pth, torchvision convnext_small under a
`self.model = convnext_small(...)` wrapper, classifier.2 -> 2 classes)
on COFW, to close the hand-over-face generalization gap documented
during evaluation (the original checkpoint was trained on a crawled
dataset skewed toward mask/sunglasses-style occlusion and did not
generalize to hand occlusion).

WHY PARTIAL FREEZING: COFW's train split is small (1,345 images, and
only ~500 of those are genuine COFW — the rest are unoccluded LFPW
faces) compared to the ~9,749 images the checkpoint was originally
trained on. Fine-tuning the whole network on that much less data risks
catastrophic forgetting of the mask/sunglasses cases it already handles
well. So by default this script freezes the early feature stages
(low-level edge/texture filters, which don't need to change) and only
fine-tunes the later stages + classifier head, at a low learning rate.
After training, evaluate on BOTH a COFW held-out set AND a few known
mask/sunglasses examples to confirm no regression — this script's
metrics alone won't catch that.

Usage:
    python fine_tune_occlusion.py \\
        --checkpoint cc/best_convnext_small.pth \\
        --labels-csv data/cofw_extracted/labels.csv \\
        --output-dir runs/occlusion_finetune \\
        --epochs 15 \\
        --freeze-until 6
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_small

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


class OcclusionModel(nn.Module):
    """
    Mirrors the checkpoint's wrapper structure (state_dict keys are
    prefixed `model.`), so `torch.load(...)['state_dict']` loads here
    with no key remapping.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = convnext_small(weights=None, num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class COFWOcclusionDataset(Dataset):
    """
    Loads face crops + binary occlusion labels from the CSV produced
    by extract_cofw.py. Crops use an expanded box around the COFW
    bbox (default +30% each side) rather than the tight bbox — a
    tight crop was found during evaluation to make occlusion even
    harder for the classifier to pick up on, likely because it crops
    out the very edges (hand/hair boundary against skin) that signal
    occlusion.
    """

    def __init__(self, csv_path: Path, split: str, expand_ratio: float = 0.3) -> None:
        df = pd.read_csv(csv_path)
        self.rows = df[df["split"] == split].reset_index(drop=True)
        self.expand_ratio = expand_ratio

    def __len__(self) -> int:
        return len(self.rows)

    def _expanded_crop(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        height, width = image.shape[:2]
        pad_w = int(w * self.expand_ratio)
        pad_h = int(h * self.expand_ratio)

        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(width, x + w + pad_w)
        y1 = min(height, y + h + pad_h)

        return image[y0:y1, x0:x1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[idx]
        image = cv2.imread(row["image_path"])
        crop = self._expanded_crop(
            image, int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
        )

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        chw = normalized.transpose(2, 0, 1).astype(np.float32)

        label = int(row["is_occluded"])
        return torch.from_numpy(chw), label


def freeze_early_stages(model: OcclusionModel, freeze_until: int) -> None:
    """
    Freeze torchvision convnext_small's `features` stages [0, freeze_until)
    (of 8 total: stage 0 is the stem, stages 1/3/5/7 are downsampling +
    residual blocks, stages 2/4/6 are the deeper blocks). Everything
    from freeze_until onward, plus the classifier head, stays trainable.
    """
    for i, stage in enumerate(model.model.features):
        if i < freeze_until:
            for param in stage.parameters():
                param.requires_grad = False


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ConvNeXt-Small on COFW occlusion labels.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_convnext_small.pth")
    parser.add_argument("--labels-csv", type=Path, required=True, help="Path to extract_cofw.py's labels.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5, help="Low LR — this is fine-tuning, not training from scratch.")
    parser.add_argument(
        "--freeze-until",
        type=int,
        default=6,
        help="Freeze features[0:freeze_until]; fine-tune the rest + classifier. 0 = fine-tune everything.",
    )
    parser.add_argument("--expand-ratio", type=float, default=0.3, help="Face bbox expansion ratio for crops.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    model = OcclusionModel()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    if args.freeze_until > 0:
        freeze_early_stages(model, args.freeze_until)

    train_dataset = COFWOcclusionDataset(args.labels_csv, split="train", expand_ratio=args.expand_ratio)
    val_dataset = COFWOcclusionDataset(args.labels_csv, split="test", expand_ratio=args.expand_ratio)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Class weighting for the train split's ~998/347 clean/occluded imbalance
    # (see labels.csv value_counts from the extraction step).
    labels = train_dataset.rows["is_occluded"].values
    class_counts = np.bincount(labels, minlength=2)
    class_weights = torch.tensor(
        [len(labels) / (2 * count) if count > 0 else 0.0 for count in class_counts],
        dtype=torch.float32,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_path = args.output_dir / "best_finetuned.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"epoch {epoch:02d}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, best_path)
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")

    print(f"\nBest val_acc: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()