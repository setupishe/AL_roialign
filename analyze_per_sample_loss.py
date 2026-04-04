import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def analyze_loss(csv_path, top_n=10):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df["total_loss"] = df["loss_iou"] + df["loss_cls"] + df["loss_dfl"]

    # Sort by epoch within each image so first/last are epoch-ordered
    df = df.sort_values(["im_file", "epoch"])

    grouped = df.groupby("im_file")["total_loss"]

    stats = pd.DataFrame({
        "mean_loss":        grouped.mean(),
        "std_loss":         grouped.std().fillna(0),
        "max_loss":         grouped.max(),
        "min_loss":         grouped.min(),
        "first_epoch_loss": grouped.first(),
        "last_epoch_loss":  grouped.last(),
        "n_epochs_seen":    grouped.count(),
    }).reset_index()

    stats["loss_drop"] = stats["first_epoch_loss"] - stats["last_epoch_loss"]

    print(f"\nTotal images tracked: {len(stats)}")
    print(f"Epoch range: {df['epoch'].min()} → {df['epoch'].max()}")

    def _show(label, subset):
        print(f"\n--- {label} ---")
        for _, row in subset.iterrows():
            name = Path(row["im_file"]).name
            print(
                f"  {name:40s}  mean={row['mean_loss']:.4f}  "
                f"first={row['first_epoch_loss']:.4f}  last={row['last_epoch_loss']:.4f}  "
                f"drop={row['loss_drop']:.4f}"
            )

    _show(f"Top {top_n} hardest (highest mean loss)",
          stats.nlargest(top_n, "mean_loss"))
    _show(f"Top {top_n} easiest (lowest mean loss)",
          stats.nsmallest(top_n, "mean_loss"))
    _show(f"Top {top_n} forgotten (loss increased most)",
          stats.nsmallest(top_n, "loss_drop"))

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze per-sample loss from YOLOv8 training")
    parser.add_argument("csv_path", type=str, help="Path to per_sample_loss.csv")
    parser.add_argument("--top", type=int, default=10, help="Number of top examples to show")
    args = parser.parse_args()

    analyze_loss(args.csv_path, top_n=args.top)
