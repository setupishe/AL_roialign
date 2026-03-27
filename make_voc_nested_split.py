#!/usr/bin/env python3
"""Build a nested VOC train list: target fraction of full train, sampled only from train_{parent}.txt.

Example (5% of full train.txt, drawn from train_0.2.txt, seed 42):
    python3 make_voc_nested_split.py --frac 0.05 --parent-frac 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--voc-root",
        type=Path,
        default=Path("/home/setupishe/datasets/VOC"),
        help="VOC dataset root (contains train.txt, train_{parent}.txt)",
    )
    p.add_argument(
        "--ultralytics-datasets",
        type=Path,
        default=Path("/home/setupishe/ultralytics/ultralytics/cfg/datasets"),
        help="Where to write VOC_{frac}.yaml",
    )
    p.add_argument("--frac", type=float, default=0.05, help="Target fraction of full train (by count)")
    p.add_argument(
        "--parent-frac",
        type=float,
        default=0.2,
        help="Pool file train_{parent}.txt (e.g. 0.2 for train_0.2.txt)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling from parent pool")
    args = p.parse_args()

    train_full = args.voc_root / "train.txt"

    def frac_tag(x: float) -> str:
        return f"{x:.4f}".rstrip("0").rstrip(".")

    frac_s = frac_tag(args.frac)
    parent_s = frac_tag(args.parent_frac)
    pool_path = args.voc_root / f"train_{parent_s}.txt"
    out_list = args.voc_root / f"train_{frac_s}.txt"

    random.seed(args.seed)

    with open(train_full) as f:
        n_full = sum(1 for _ in f)

    with open(pool_path) as f:
        pool = [line for line in f if line.strip()]

    n_target = max(1, round(args.frac * n_full))
    if n_target > len(pool):
        raise SystemExit(
            f"need {n_target} images (frac * full) but {pool_path} has only {len(pool)}"
        )

    chosen = random.sample(pool, n_target)
    chosen.sort(key=pool.index)

    out_list.write_text("".join(chosen))
    print(f"wrote {out_list} ({len(chosen)} lines, {args.frac:.4f} * {n_full} ≈ {n_target})")

    base_yaml = args.ultralytics_datasets / "VOC.yaml"
    out_yaml = args.ultralytics_datasets / f"VOC_{frac_s}.yaml"
    text = base_yaml.read_text()
    if "train: train.txt" not in text:
        raise SystemExit(f"expected 'train: train.txt' in {base_yaml}")
    out_yaml.write_text(text.replace("train: train.txt", f"train: train_{frac_s}.txt", 1))
    print(f"wrote {out_yaml}")


if __name__ == "__main__":
    main()
