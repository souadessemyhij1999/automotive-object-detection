"""
Fine-tune a pretrained YOLO26 model on the KITTI driving dataset
(German street scenes recorded around Karlsruhe).

The dataset (~390 MB) downloads automatically the first time you run this.
Use a GPU (Google Colab is free) - on a laptop CPU training takes many hours.

Examples:
    python src/train.py                         # full run, 50 epochs
    python src/train.py --epochs 1 --fraction 0.05   # quick test that everything works
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train the automotive detector.")
    p.add_argument("--data", default="configs/kitti.yaml", help="dataset config")
    p.add_argument("--model", default="yolo26n.pt", help="starting weights (n=nano, s=small, m=medium)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640, help="training image size in pixels")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fraction", type=float, default=1.0, help="share of training images to use (0-1)")
    p.add_argument("--device", default=None, help="0 for first GPU, 'cpu' for CPU (default: auto)")
    p.add_argument("--name", default="kitti_yolo26n", help="run name under runs/detect/")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=args.device,
        project=str(Path("runs/detect").resolve()),
        name=args.name,
        exist_ok=True,
        patience=15,   # stop early if validation mAP stops improving
        seed=42,       # reproducible results
        plots=True,    # saves confusion matrix, PR curves, sample predictions
    )

    best = Path(model.trainer.best)
    Path("weights").mkdir(exist_ok=True)
    shutil.copy(best, "weights/best.pt")
    print(f"\nTraining finished. Best weights copied to weights/best.pt")
    print(f"Training plots are in {best.parent.parent}/")


if __name__ == "__main__":
    main()
