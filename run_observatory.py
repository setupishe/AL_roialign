"""Run observatory analysis and print results (non-interactive version of the notebook)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from observatory_utils import (
    VOC_CLASSES,
    STRATEGY_CHAINS_20_70,
    check_chain_monotonicity,
    class_dist_for_chain,
    class_distribution,
    diff_splits,
    full_pool_class_distribution,
    get_chain_additions,
    get_chain_splits,
    images_without_annotations,
    jaccard,
    object_size_stats,
    overlap_matrix_np,
    parse_split,
)


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ── 1. Chain structure ──────────────────────────────────────────────
section("1. CHAIN STRUCTURE — Images per fraction")

for name, chain in STRATEGY_CHAINS_20_70.items():
    splits = get_chain_splits(chain)
    if not splits:
        print(f"  {name}: no split files found")
        continue
    print(f"--- {name} ---")
    fracs = sorted(splits.keys())
    prev_set = None
    for f in fracs:
        cur_set = parse_split(splits[f])
        added = len(cur_set - prev_set) if prev_set is not None else len(cur_set)
        label = "(initial)" if prev_set is None else f"(+{added})"
        print(f"  {f:.0%}: {len(cur_set):>5} images {label}")
        prev_set = cur_set
    print()


# ── 1b. Monotonicity check ─────────────────────────────────────────
section("1b. MONOTONICITY CHECK — Are selections strictly additive?")

for name, chain in STRATEGY_CHAINS_20_70.items():
    mono = check_chain_monotonicity(chain)
    if not mono:
        continue
    all_mono = all(v["monotonic"] for v in mono.values())
    status = "OK" if all_mono else "NON-MONOTONIC"
    print(f"--- {name} [{status}] ---")
    for step, info in mono.items():
        flag = "" if info["monotonic"] else " <<<< DROPPED IMAGES"
        print(f"  {step}: +{info['added']} added, -{info['dropped']} dropped, "
              f"{info['prev_size']}→{info['next_size']}{flag}")
    print()


# ── 2. Selection overlap ───────────────────────────────────────────
section("2. SELECTION OVERLAP — Jaccard on ADDED images")

fraction_pairs = [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7)]

for prev_f, next_f in fraction_pairs:
    step_label = f"{prev_f}→{next_f}"
    additions_by_strategy = {}

    for name, chain in STRATEGY_CHAINS_20_70.items():
        splits = get_chain_splits(chain)
        if prev_f in splits and next_f in splits:
            added = diff_splits(splits[prev_f], splits[next_f])
            additions_by_strategy[name] = added

    if len(additions_by_strategy) < 2:
        continue

    print(f"--- Step {step_label} ---")
    names = list(additions_by_strategy.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            j = jaccard(additions_by_strategy[n1], additions_by_strategy[n2])
            print(f"  {n1:30s} vs {n2:30s}  Jaccard = {j:.4f}")
    print()


# ── 3. Class distribution ──────────────────────────────────────────
section("3. CLASS DISTRIBUTION — Pool baseline")

pool_dist = full_pool_class_distribution()
total_annotations = sum(pool_dist.values())
pool_fracs = {cls: count / total_annotations for cls, count in pool_dist.items()}

for cls_id in sorted(pool_dist.keys()):
    print(f"  {VOC_CLASSES[cls_id]:>15}: {pool_dist[cls_id]:>5} ({pool_fracs[cls_id]:.1%})")
print(f"  {'TOTAL':>15}: {total_annotations}")


# ── 4. Class selection bias ─────────────────────────────────────────
section("4. CLASS SELECTION BIAS — ratio vs pool baseline")

strategies_to_compare = ["random", "distance", "distance_matryoshka_m", "density"]
bias_data = []

for strategy_name in strategies_to_compare:
    chain = STRATEGY_CHAINS_20_70.get(strategy_name)
    if chain is None:
        continue
    splits = get_chain_splits(chain)
    fracs_available = sorted(splits.keys())

    for i in range(len(fracs_available) - 1):
        prev_f, next_f = fracs_available[i], fracs_available[i + 1]
        added = diff_splits(splits[prev_f], splits[next_f])
        dist = class_distribution(added)
        total = sum(dist.values())
        if total == 0:
            continue

        for cls_id in VOC_CLASSES:
            sel_frac = dist.get(cls_id, 0) / total
            pool_frac = pool_fracs.get(cls_id, 1e-9)
            bias_data.append({
                "strategy": strategy_name,
                "step": f"{prev_f}→{next_f}",
                "class": VOC_CLASSES[cls_id],
                "class_id": cls_id,
                "bias": sel_frac / pool_frac,
            })

bias_df = pd.DataFrame(bias_data)
avg_bias = bias_df.groupby(["strategy", "class"])["bias"].mean().unstack(level=0)
print(avg_bias.to_string())


# ── 5. Distance vs Matryoshka bias diff ─────────────────────────────
section("5. MATRYOSHKA vs VANILLA DISTANCE — Class bias difference")

if "distance" in avg_bias.columns and "distance_matryoshka_m" in avg_bias.columns:
    diff = (avg_bias["distance_matryoshka_m"] - avg_bias["distance"]).sort_values()
    print("Positive = matryoshka selects MORE of this class than vanilla distance")
    print()
    for cls_name, val in diff.items():
        marker = ">>>" if abs(val) > 0.2 else "   "
        print(f"  {marker} {cls_name:>15}: {val:+.4f}")
else:
    print("Missing data for comparison")


# ── 6. Unique selections ───────────────────────────────────────────
section("6. STRATEGY SIGNATURES — Unique selections")

key_strategies = ["distance", "distance_matryoshka_m", "random"]
all_additions = {}

for name in key_strategies:
    chain = STRATEGY_CHAINS_20_70.get(name)
    if chain is None:
        continue
    adds = get_chain_additions(chain)
    combined = set()
    for step_adds in adds.values():
        combined |= step_adds
    all_additions[name] = combined

if "distance" in all_additions and "distance_matryoshka_m" in all_additions:
    only_vanilla = all_additions["distance"] - all_additions["distance_matryoshka_m"]
    only_matryoshka = all_additions["distance_matryoshka_m"] - all_additions["distance"]
    shared = all_additions["distance"] & all_additions["distance_matryoshka_m"]

    print(f"Total distance-only selections:    {len(only_vanilla)}")
    print(f"Total matryoshka-only selections:   {len(only_matryoshka)}")
    print(f"Shared selections:                  {len(shared)}")
    print()

    dist_vanilla = class_distribution(only_vanilla)
    dist_matr = class_distribution(only_matryoshka)

    print(f"{'Class':>15}  {'Vanilla-only':>12}  {'Matryoshka-only':>15}")
    print(f"{'-----':>15}  {'------------':>12}  {'---------------':>15}")
    for cls_id in sorted(VOC_CLASSES.keys()):
        cn = VOC_CLASSES[cls_id]
        print(f"  {cn:>13}  {dist_vanilla.get(cls_id, 0):>12}  {dist_matr.get(cls_id, 0):>15}")


# ── 7. Background image ratio ──────────────────────────────────────
section("7. BACKGROUND IMAGE RATIO")

bg_data = []
for strategy_name in STRATEGY_CHAINS_20_70:
    chain = STRATEGY_CHAINS_20_70[strategy_name]
    splits = get_chain_splits(chain)
    for frac, path in sorted(splits.items()):
        all_stems = parse_split(path)
        bg = images_without_annotations(all_stems)
        bg_data.append({
            "strategy": strategy_name,
            "fraction": f"{frac:.0%}",
            "total": len(all_stems),
            "bg": len(bg),
            "bg_ratio": f"{len(bg)/len(all_stems):.1%}" if all_stems else "N/A",
        })

bg_df = pd.DataFrame(bg_data)
print(bg_df.to_string(index=False))


# ── 8. Object size comparison ──────────────────────────────────────
section("8. OBJECT SIZE — Mean area by strategy per step")

for prev_f, next_f in [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]:
    step_label = f"{prev_f}→{next_f}"
    print(f"--- Step {step_label} ---")
    for strategy_name in strategies_to_compare:
        chain = STRATEGY_CHAINS_20_70.get(strategy_name)
        if chain is None:
            continue
        splits = get_chain_splits(chain)
        if prev_f not in splits or next_f not in splits:
            continue
        added = diff_splits(splits[prev_f], splits[next_f])
        sizes = object_size_stats(added)
        all_areas = [a for areas in sizes.values() for a in areas]
        if all_areas:
            print(f"  {strategy_name:30s}  mean={np.mean(all_areas):.4f}  median={np.median(all_areas):.4f}  n={len(all_areas)}")
    print()


# ── 9. Summary table ───────────────────────────────────────────────
section("9. SUMMARY TABLE")

summary_rows = []
for strategy_name in strategies_to_compare:
    chain = STRATEGY_CHAINS_20_70.get(strategy_name)
    if chain is None:
        continue
    additions = get_chain_additions(chain)
    for step, stems in additions.items():
        dist = class_distribution(stems)
        sizes = object_size_stats(stems)
        all_areas = [a for areas in sizes.values() for a in areas]
        bg = images_without_annotations(stems)
        summary_rows.append({
            "Strategy": strategy_name,
            "Step": step,
            "Imgs": len(stems),
            "Annots": sum(dist.values()),
            "Classes": len(dist),
            "BG imgs": len(bg),
            "Mean area": f"{np.mean(all_areas):.4f}" if all_areas else "N/A",
        })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

print("\n\nDone.")
