"""
Turn the detections.csv written by detect.py into a short traffic report:
object counts per class and (for videos) objects per frame over time.

    python src/summarize_detections.py --csv results/detections/detections.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/detections/detections.csv")
    args = p.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("No detections in the CSV.")

    counts = df["class"].value_counts()
    print("Objects detected per class:\n" + counts.to_string())
    print(f"\nAverage confidence: {df['confidence'].mean():.2f}")

    fig, axes = plt.subplots(1, 2 if df["frame"].nunique() > 1 else 1, figsize=(11, 4))
    axes = axes if hasattr(axes, "__len__") else [axes]

    counts.plot.barh(ax=axes[0], color="#1c69d4")
    axes[0].invert_yaxis()
    axes[0].set_title("Detections per class")
    axes[0].set_xlabel("count")

    if len(axes) > 1:
        per_frame = df.groupby(["frame", "class"]).size().unstack(fill_value=0)
        per_frame.rolling(10, min_periods=1).mean().plot(ax=axes[1])
        axes[1].set_title("Objects in view over time (10-frame average)")
        axes[1].set_xlabel("frame")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = csv_path.parent / "detection_summary.png"
    fig.savefig(out, dpi=150)
    print(f"\nChart saved to {out}")


if __name__ == "__main__":
    main()
