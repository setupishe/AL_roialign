#!/usr/bin/env python3
"""Generate observatory visualizations and save PNGs (run from bel_conf with .venv)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from observatory_utils import (
    VOC_CLASSES,
    STRATEGY_CHAINS_20_70,
    check_chain_monotonicity,
    class_distribution,
    diff_splits,
    full_pool_class_distribution,
    get_chain_additions,
    get_chain_splits,
    images_without_annotations,
    object_size_stats,
    overlap_matrix_np,
    parse_split,
)

OUTPUT_DIR = Path("/home/setupishe/LBC/phd/figures/experiment_observatory")
DPI = 150
STRATEGIES_COMPARE = ["random", "distance", "distance_matryoshka_m", "density"]


def ensure_out():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_chain_sizes():
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, chain in STRATEGY_CHAINS_20_70.items():
        splits = get_chain_splits(chain)
        if not splits:
            continue
        fracs = sorted(splits.keys())
        counts = [len(parse_split(splits[f])) for f in fracs]
        ax.plot(fracs, counts, marker="o", label=name)
    ax.set_xlabel("Labeled fraction")
    ax.set_ylabel("Number of images in train split")
    ax.set_title("VOC train split sizes by strategy")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, "01_chain_sizes.png")


def fig_jaccard_added_all_steps():
    fraction_pairs = [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    for idx, (prev_f, next_f) in enumerate(fraction_pairs):
        ax = axes[idx]
        additions = {}
        for name, chain in STRATEGY_CHAINS_20_70.items():
            splits = get_chain_splits(chain)
            if prev_f in splits and next_f in splits:
                additions[name] = diff_splits(splits[prev_f], splits[next_f])
        if len(additions) < 2:
            ax.axis("off")
            continue
        mat, labels = overlap_matrix_np(additions)
        sns.heatmap(
            mat,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"shrink": 0.6},
        )
        ax.set_title(f"Jaccard on ADDED images: {prev_f}→{next_f}")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    axes[-1].axis("off")
    fig.suptitle("Selection overlap between strategies (per AL step)", fontsize=12, y=1.02)
    plt.tight_layout()
    save(fig, "02_jaccard_added_per_step.png")


def fig_jaccard_cumulative():
    fracs_target = [0.3, 0.4, 0.5, 0.6, 0.7]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    for idx, frac in enumerate(fracs_target):
        ax = axes[idx]
        sets_at = {}
        for name, chain in STRATEGY_CHAINS_20_70.items():
            splits = get_chain_splits(chain)
            if frac in splits:
                sets_at[name] = parse_split(splits[frac])
        if len(sets_at) < 2:
            ax.axis("off")
            continue
        mat, labels = overlap_matrix_np(sets_at)
        sns.heatmap(
            mat,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"shrink": 0.6},
        )
        ax.set_title(f"Cumulative set Jaccard @ {frac:.0%}")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    axes[-1].axis("off")
    fig.suptitle("Full train-list overlap at each labeled fraction", fontsize=12, y=1.02)
    plt.tight_layout()
    save(fig, "03_jaccard_cumulative_sets.png")


def fig_class_bias_heatmap():
    pool_dist = full_pool_class_distribution()
    total = sum(pool_dist.values())
    pool_fracs = {c: pool_dist[c] / total for c in pool_dist}

    bias_rows = []
    for strategy_name in STRATEGIES_COMPARE:
        chain = STRATEGY_CHAINS_20_70.get(strategy_name)
        if not chain:
            continue
        splits = get_chain_splits(chain)
        fracs = sorted(splits.keys())
        for i in range(len(fracs) - 1):
            prev_f, next_f = fracs[i], fracs[i + 1]
            added = diff_splits(splits[prev_f], splits[next_f])
            dist = class_distribution(added)
            tot = sum(dist.values())
            if tot == 0:
                continue
            for cls_id in VOC_CLASSES:
                sel_frac = dist.get(cls_id, 0) / tot
                pool_frac = pool_fracs.get(cls_id, 1e-9)
                bias_rows.append(
                    {
                        "strategy": strategy_name,
                        "class": VOC_CLASSES[cls_id],
                        "bias": sel_frac / pool_frac,
                    }
                )

    df = pd.DataFrame(bias_rows)
    avg_bias = df.groupby(["class", "strategy"])["bias"].mean().unstack(level=1)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        avg_bias,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=1.0,
        ax=ax,
        linewidths=0.3,
    )
    ax.set_title("Mean class selection bias vs full pool\n(>1 = over-selected in additions)")
    ax.set_ylabel("VOC class")
    plt.tight_layout()
    save(fig, "04_class_bias_heatmap.png")


def fig_matryoshka_vs_distance():
    pool_dist = full_pool_class_distribution()
    total = sum(pool_dist.values())
    pool_fracs = {c: pool_dist[c] / total for c in pool_dist}

    bias_rows = []
    for strategy_name in ["distance", "distance_matryoshka_m"]:
        chain = STRATEGY_CHAINS_20_70[strategy_name]
        splits = get_chain_splits(chain)
        fracs = sorted(splits.keys())
        for i in range(len(fracs) - 1):
            prev_f, next_f = fracs[i], fracs[i + 1]
            added = diff_splits(splits[prev_f], splits[next_f])
            dist = class_distribution(added)
            tot = sum(dist.values())
            if tot == 0:
                continue
            for cls_id in VOC_CLASSES:
                sel_frac = dist.get(cls_id, 0) / tot
                pool_frac = pool_fracs.get(cls_id, 1e-9)
                bias_rows.append(
                    {
                        "strategy": strategy_name,
                        "class": VOC_CLASSES[cls_id],
                        "bias": sel_frac / pool_frac,
                    }
                )

    df = pd.DataFrame(bias_rows)
    avg = df.groupby(["class", "strategy"])["bias"].mean().unstack(level=1)
    if "distance" not in avg.columns or "distance_matryoshka_m" not in avg.columns:
        return
    diff = (avg["distance_matryoshka_m"] - avg["distance"]).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#c62828" if v < 0 else "#2e7d32" for v in diff.values]
    ax.barh(range(len(diff)), diff.values, color=colors, height=0.7)
    ax.set_yticks(range(len(diff)))
    ax.set_yticklabels(diff.index, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Bias difference (matryoshka − vanilla distance)")
    ax.set_title("Class-level shift: distance+Matryoshka vs vanilla distance AL")
    plt.tight_layout()
    save(fig, "05_matryoshka_vs_distance_class_shift.png")


def fig_unique_selections():
    all_add = {}
    for name in ["distance", "distance_matryoshka_m", "random"]:
        chain = STRATEGY_CHAINS_20_70.get(name)
        if not chain:
            continue
        combined = set()
        for stems in get_chain_additions(chain).values():
            combined |= stems
        all_add[name] = combined

    if "distance" not in all_add or "distance_matryoshka_m" not in all_add:
        return

    only_v = all_add["distance"] - all_add["distance_matryoshka_m"]
    only_m = all_add["distance_matryoshka_m"] - all_add["distance"]
    dv = class_distribution(only_v)
    dm = class_distribution(only_m)

    classes = sorted(VOC_CLASSES.keys())
    names = [VOC_CLASSES[c] for c in classes]
    x = np.arange(len(classes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(
        x - w / 2,
        [dv.get(c, 0) for c in classes],
        w,
        label=f"Only vanilla distance ({len(only_v)} imgs)",
        color="#1565c0",
    )
    ax.bar(
        x + w / 2,
        [dm.get(c, 0) for c in classes],
        w,
        label=f"Only matryoshka distance ({len(only_m)} imgs)",
        color="#ef6c00",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Annotation count in unique selections")
    ax.set_title(
        "Annotations on images selected by one distance strategy but not the other (all steps combined)"
    )
    ax.legend()
    plt.tight_layout()
    save(fig, "06_unique_selections_by_class.png")


def fig_monotonicity():
    rows = []
    for name, chain in STRATEGY_CHAINS_20_70.items():
        for step, info in check_chain_monotonicity(chain).items():
            rows.append(
                {
                    "strategy": name,
                    "step": step,
                    "dropped": info["dropped"],
                    "added": info["added"],
                    "monotonic": info["monotonic"],
                }
            )
    df = pd.DataFrame(rows)
    df_nonzero = df[df["dropped"] > 0]
    if df_nonzero.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "All chains monotonic (no dropped images)", ha="center", va="center")
        ax.axis("off")
        save(fig, "07_monotonicity_dropped.png")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df.pivot(index="step", columns="strategy", values="dropped").fillna(0)
    pivot.plot(kind="bar", ax=ax, rot=15)
    ax.set_ylabel("Images dropped vs previous step")
    ax.set_title("Non-monotonic AL splits: images removed when advancing fraction")
    ax.legend(title="strategy", fontsize=8)
    plt.tight_layout()
    save(fig, "07_monotonicity_dropped.png")


def fig_object_sizes():
    steps = [(0.2, 0.3), (0.3, 0.4)]
    fig, axes = plt.subplots(len(steps), 2, figsize=(14, 5 * len(steps)))

    for row, (prev_f, next_f) in enumerate(steps):
        ax_hist, ax_box = axes[row, 0], axes[row, 1]
        for strategy_name in STRATEGIES_COMPARE:
            chain = STRATEGY_CHAINS_20_70.get(strategy_name)
            if not chain:
                continue
            splits = get_chain_splits(chain)
            if prev_f not in splits or next_f not in splits:
                continue
            added = diff_splits(splits[prev_f], splits[next_f])
            sizes = object_size_stats(added)
            areas = [a for lst in sizes.values() for a in lst]
            if areas:
                ax_hist.hist(
                    areas, bins=40, alpha=0.35, label=strategy_name, density=True, range=(0, 0.35)
                )

        ax_hist.set_xlim(0, 0.35)
        ax_hist.set_xlabel("Object area (norm w×h)")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"Object size in ADDED images: {prev_f}→{next_f}")
        ax_hist.legend(fontsize=7)

        rows_box = []
        for strategy_name in STRATEGIES_COMPARE:
            chain = STRATEGY_CHAINS_20_70.get(strategy_name)
            if not chain:
                continue
            splits = get_chain_splits(chain)
            if prev_f not in splits or next_f not in splits:
                continue
            added = diff_splits(splits[prev_f], splits[next_f])
            sizes = object_size_stats(added)
            for a in [x for lst in sizes.values() for x in lst]:
                rows_box.append({"strategy": strategy_name, "area": a})
        if rows_box:
            bdf = pd.DataFrame(rows_box)
            sns.boxplot(data=bdf, x="strategy", y="area", ax=ax_box, showfliers=False)
            ax_box.set_ylim(0, 0.25)
            ax_box.set_title(f"Area distribution by strategy ({prev_f}→{next_f})")
            ax_box.tick_params(axis="x", rotation=20, labelsize=8)

    plt.tight_layout()
    save(fig, "08_object_size_added_images.png")


def fig_background_ratio():
    rows = []
    for strategy_name in STRATEGY_CHAINS_20_70:
        chain = STRATEGY_CHAINS_20_70[strategy_name]
        for frac, path in sorted(get_chain_splits(chain).items()):
            stems = parse_split(path)
            bg = images_without_annotations(stems)
            rows.append(
                {
                    "strategy": strategy_name,
                    "fraction": frac,
                    "bg_ratio": len(bg) / len(stems) if stems else 0,
                }
            )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in df["strategy"].unique():
        sub = df[df["strategy"] == name]
        ax.plot(sub["fraction"], sub["bg_ratio"], marker="o", label=name)
    ax.set_xlabel("Labeled fraction")
    ax.set_ylabel("Fraction of images with no GT boxes")
    ax.set_title("Background (empty-label) images in train splits")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, "09_background_ratio.png")


def write_readme():
    readme = OUTPUT_DIR / "README.md"
    readme.write_text(
        """# Experiment observatory figures

Auto-generated PNGs from `bel_conf/generate_observatory_figures.py`.

- `01_chain_sizes.png` — Train list size per strategy vs fraction
- `02_jaccard_added_per_step.png` — Pairwise Jaccard on **newly added** images each step
- `03_jaccard_cumulative_sets.png` — Jaccard on full train lists at each fraction
- `04_class_bias_heatmap.png` — Mean selection bias vs VOC pool (per class × strategy)
- `05_matryoshka_vs_distance_class_shift.png` — Matryoshka minus vanilla distance bias
- `06_unique_selections_by_class.png` — Class counts for strategy-unique image picks
- `07_monotonicity_dropped.png` — Images dropped when advancing fraction (if any)
- `08_object_size_added_images.png` — Normalized object area in added images
- `09_background_ratio.png` — Share of images with no annotations

Re-run: `cd /home/setupishe/bel_conf && .venv/bin/python generate_observatory_figures.py`
"""
    )
    print(f"  wrote {readme}")


def main():
    ensure_out()
    sns.set_theme(style="whitegrid", font_scale=1.0)
    print(f"Writing to {OUTPUT_DIR}/")
    fig_chain_sizes()
    fig_jaccard_added_all_steps()
    fig_jaccard_cumulative()
    fig_class_bias_heatmap()
    fig_matryoshka_vs_distance()
    fig_unique_selections()
    fig_monotonicity()
    fig_object_sizes()
    fig_background_ratio()
    write_readme()
    print("Done.")


if __name__ == "__main__":
    main()
