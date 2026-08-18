"""
Export a fine-tuned checkpoint (from fine_tune_occlusion.py) to ONNX,
matching the exact contract app/quality/semantic/occlusion.py's
ModelOcclusionDetector already expects — so the new .onnx is a
drop-in replacement, no code changes needed elsewhere:

  Input:  [1, 3, 224, 224] float32, RGB, scaled to [0,1] then
          ImageNet-normalized, NCHW.
  Output: [1, 2] logits — softmax index 1 = occluded.

Usage:
    python export_occlusion_onnx.py \\
        --checkpoint runs/occlusion_finetune/best_finetuned.pth \\
        --output models/face_occlusion_finetuned.onnx
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import convnext_small

INPUT_SIZE = 224


class OcclusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = convnext_small(weights=None, num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fine-tuned occlusion checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=12, help="Matches the original export's opset version.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = OcclusionModel()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(args.output),
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes=None,  # fixed batch size of 1, matching ModelOcclusionDetector's usage
    )

    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()