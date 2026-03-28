#!/usr/bin/env python3
"""Build a nested train split for any dataset.

The target split contains `frac * full_train_count` items, sampled only from the
parent split file.

Examples:
    python3 make_voc_nested_split.py --frac 0.05 --parent-frac 0.2
    python3 make_voc_nested_split.py --dataset-name COCO --dataset-root /path/to/coco \\
        --train-file train2017.txt --frac 0.2 --parent-frac 0.5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def frac_tag(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


def split_path(dataset_root: Path, train_file: str, frac: float) -> Path:
    train_path = Path(train_file)
    return dataset_root / f"{train_path.stem}_{frac_tag(frac)}{train_path.suffix}"


def rewrite_train_entry(text: str, train_file: str, out_train_file: str) -> str:
    for line in text.splitlines():
        if not line.startswith("train:"):
            continue

        tail = line.split(":", 1)[1]
        value, _, comment = tail.partition("#")
        value = value.strip()
        quote = value[0] if value[:1] in {"'", '"'} and value[-1:] == value[:1] else ""
        raw_path = value[1:-1] if quote else value

        if Path(raw_path).name != train_file:
            continue

        parent = Path(raw_path).parent
        new_path = str(parent / out_train_file) if str(parent) != "." else out_train_file
        replacement = f"train: {quote}{new_path}{quote}"
        if comment:
            replacement = f"{replacement} #{comment}"
        return text.replace(line, replacement, 1)

    raise SystemExit(f"expected train entry ending with '{train_file}'")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/setupishe/datasets/VOC"),
        help="Dataset root that contains the full train list and nested split lists",
    )
    p.add_argument(
        "--ultralytics-datasets",
        type=Path,
        default=Path("/home/setupishe/ultralytics/ultralytics/cfg/datasets"),
        help="Directory where dataset YAML files live",
    )
    p.add_argument(
        "--dataset-name",
        default="VOC",
        help="Dataset YAML stem, e.g. VOC -> VOC.yaml, COCO -> COCO.yaml",
    )
    p.add_argument(
        "--train-file",
        default="train.txt",
        help="Full train list filename inside dataset root, used for total count",
    )
    p.add_argument(
        "--split-base-file",
        default=None,
        help="Base filename for parent/output split files, defaults to --train-file",
    )
    p.add_argument(
        "--yaml-train-file",
        default=None,
        help="Train filename currently referenced by the base YAML, defaults to --train-file",
    )
    p.add_argument(
        "--base-yaml",
        default=None,
        help="Base YAML filename to copy from, defaults to {dataset-name}.yaml",
    )
    p.add_argument(
        "--voc-root",
        dest="dataset_root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--frac", type=float, default=0.05, help="Target fraction of full train (by count)")
    p.add_argument(
        "--parent-frac",
        type=float,
        default=0.2,
        help="Pool split fraction used to build the nested subset",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling from parent pool")
    args = p.parse_args()

    split_base_file = args.split_base_file or args.train_file
    yaml_train_file = args.yaml_train_file or args.train_file
    base_yaml_name = args.base_yaml or f"{args.dataset_name}.yaml"

    train_full = args.dataset_root / args.train_file
    frac_s = frac_tag(args.frac)
    pool_path = split_path(args.dataset_root, split_base_file, args.parent_frac)
    out_list = split_path(args.dataset_root, split_base_file, args.frac)

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
    pool_order = {line: idx for idx, line in enumerate(pool)}
    chosen.sort(key=pool_order.__getitem__)

    out_list.write_text("".join(chosen))
    print(f"wrote {out_list} ({len(chosen)} lines, {args.frac:.4f} * {n_full} ≈ {n_target})")

    base_yaml = args.ultralytics_datasets / base_yaml_name
    out_yaml = args.ultralytics_datasets / f"{args.dataset_name}_{frac_s}.yaml"
    text = base_yaml.read_text()
    replaced = rewrite_train_entry(text, yaml_train_file, out_list.name)
    out_yaml.write_text(replaced)
    print(f"wrote {out_yaml}")


if __name__ == "__main__":
    main()
