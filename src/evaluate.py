"""
Evaluate a model on the validation set and export the results in a
business-friendly format:

  results/metrics_per_class.csv   per-class precision, recall, mAP (open in Excel / Power BI)
  results/metrics_per_class.png   bar chart for the README
  results/summary.md              markdown table to paste into the README

Optionally compares your fine-tuned model against the pretrained baseline.

Examples:
    python src/evaluate.py
    python src/evaluate.py --model weights/best.pt --data configs/kitti.yaml
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the automotive detector.")
    p.add_argument("--model", default="weights/best.pt")
    p.add_argument("--data", default="configs/kitti.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device, plots=True,
                        project=str(Path("runs/detect").resolve()), name="eval", exist_ok=True)
    box = metrics.box

    rows = []
    for i, cls_id in enumerate(box.ap_class_index):
        rows.append({
            "class": model.names[int(cls_id)],
            "precision": round(float(box.p[i]), 3),
            "recall": round(float(box.r[i]), 3),
            "mAP50": round(float(box.ap50[i]), 3),
            "mAP50-95": round(float(box.ap[i]), 3),
        })
    df = pd.DataFrame(rows).sort_values("mAP50", ascending=False)
    df.to_csv(out / "metrics_per_class.csv", index=False)

    # Chart: mAP50 per class
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(df["class"], df["mAP50"], color="#1c69d4")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("mAP@0.5 (higher is better)")
    ax.set_title("Detection accuracy per class (KITTI validation set)")
    for y, v in enumerate(df["mAP50"]):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "metrics_per_class.png", dpi=150)

    speed = metrics.speed  # ms per image
    summary = [
        f"**Model:** `{args.model}`  |  **Data:** `{args.data}`  |  **Image size:** {args.imgsz}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| mAP@0.5 | {box.map50:.3f} |",
        f"| mAP@0.5:0.95 | {box.map:.3f} |",
        f"| Precision | {box.mp:.3f} |",
        f"| Recall | {box.mr:.3f} |",
        f"| Inference time | {speed['inference']:.1f} ms / image |",
        "",
        "| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |",
        "|---|---|---|---|---|",
    ]
    summary += [f"| {r['class']} | {r['precision']} | {r['recall']} | {r['mAP50']} | {r['mAP50-95']} |"
                for r in df.to_dict("records")]
    (out / "summary.md").write_text("\n".join(summary) + "\n")

    print("\n".join(summary))
    print(f"\nSaved: {out}/metrics_per_class.csv, metrics_per_class.png, summary.md")


if __name__ == "__main__":
    main()
