"""Experiment observatory utilities for AL selection and class distribution analysis."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

VOC_CLASSES = {
    0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
    5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow",
    10: "diningtable", 11: "dog", 12: "horse", 13: "motorbike", 14: "person",
    15: "pottedplant", 16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor",
}

DATASETS_DIR = Path("/home/setupishe/datasets/VOC")
LABELS_DIR = DATASETS_DIR / "labels" / "train"
RUNS_DIR = Path("/home/setupishe/ultralytics/runs/detect")


# ---------------------------------------------------------------------------
# Split file parsing
# ---------------------------------------------------------------------------

def parse_split(path: str | Path) -> set[str]:
    """Read a train_*.txt and return set of image stems (no extension, no path prefix)."""
    stems = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stem = Path(line).stem
            stems.add(stem)
    return stems


def diff_splits(prev_path: str | Path, next_path: str | Path) -> set[str]:
    """Return image stems that were ADDED going from prev to next split."""
    return parse_split(next_path) - parse_split(prev_path)


# ---------------------------------------------------------------------------
# Label reading
# ---------------------------------------------------------------------------

def read_image_labels(stem: str, labels_dir: Path = LABELS_DIR) -> list[tuple[int, list[float]]]:
    """Read YOLO label file for an image stem. Returns [(class_id, [cx, cy, w, h]), ...]."""
    label_path = labels_dir / f"{stem}.txt"
    if not label_path.exists():
        return []
    annotations = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                annotations.append((cls_id, bbox))
    return annotations


def class_distribution(stems: set[str], labels_dir: Path = LABELS_DIR) -> Counter:
    """Count annotations per class across a set of image stems."""
    counts = Counter()
    for stem in stems:
        for cls_id, _ in read_image_labels(stem, labels_dir):
            counts[cls_id] += 1
    return counts


def images_without_annotations(stems: set[str], labels_dir: Path = LABELS_DIR) -> set[str]:
    """Return stems that have no annotations (background images)."""
    return {s for s in stems if not read_image_labels(s, labels_dir)}


def object_size_stats(stems: set[str], labels_dir: Path = LABELS_DIR) -> dict[int, list[float]]:
    """Return {class_id: [areas]} where area = w*h in normalized coords."""
    sizes: dict[int, list[float]] = {}
    for stem in stems:
        for cls_id, bbox in read_image_labels(stem, labels_dir):
            sizes.setdefault(cls_id, []).append(bbox[2] * bbox[3])
    return sizes


# ---------------------------------------------------------------------------
# Overlap / similarity
# ---------------------------------------------------------------------------

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def overlap_matrix(sets: dict[str, set]) -> dict[tuple[str, str], float]:
    """Compute pairwise Jaccard similarity between named sets."""
    names = list(sets.keys())
    result = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j >= i:
                result[(n1, n2)] = jaccard(sets[n1], sets[n2])
                result[(n2, n1)] = result[(n1, n2)]
    return result


def overlap_matrix_np(sets: dict[str, set]) -> tuple[np.ndarray, list[str]]:
    """Return (matrix, labels) for heatmap plotting."""
    names = list(sets.keys())
    n = len(names)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = jaccard(sets[names[i]], sets[names[j]])
    return mat, names


# ---------------------------------------------------------------------------
# Strategy chain definitions
# ---------------------------------------------------------------------------

def _split_path(fraction: float, suffix: str = "") -> Path:
    if suffix:
        return DATASETS_DIR / f"train_{fraction}_{suffix}.txt"
    return DATASETS_DIR / f"train_{fraction}.txt"


STRATEGY_CHAINS_20_70 = {
    "random": {
        "fractions": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "suffix": "",
    },
    "distance": {
        "fractions": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "suffix": "distance",
        "base_fraction": 0.2,
    },
    "distance_matryoshka_m": {
        "fractions": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "suffix": "distance_matryoshka_everything_really_everything_m",
        "base_fraction": 0.2,
    },
    "density": {
        "fractions": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "suffix": "density",
        "base_fraction": 0.2,
    },
    "confidences": {
        "fractions": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "suffix": "confidences",
        "base_fraction": 0.2,
    },
}


def get_chain_splits(chain: dict) -> dict[float, Path]:
    """Return {fraction: Path} for a strategy chain, only including existing files."""
    result = {}
    base_frac = chain.get("base_fraction")
    suffix = chain["suffix"]
    for frac in chain["fractions"]:
        if base_frac is not None and frac <= base_frac:
            p = _split_path(frac, "")
        else:
            p = _split_path(frac, suffix)
        if p.exists():
            result[frac] = p
    return result


def get_chain_additions(chain: dict) -> dict[str, set[str]]:
    """Return {'{prev}->{next}': added_stems} for each consecutive pair in the chain."""
    splits = get_chain_splits(chain)
    fracs = sorted(splits.keys())
    additions = {}
    for i in range(len(fracs) - 1):
        prev_f, next_f = fracs[i], fracs[i + 1]
        key = f"{prev_f}→{next_f}"
        additions[key] = diff_splits(splits[prev_f], splits[next_f])
    return additions


def check_chain_monotonicity(chain: dict) -> dict[str, dict]:
    """Check if each step is a strict superset of the previous step.

    Returns {step: {'monotonic': bool, 'dropped': int, 'added': int}} for each step.
    """
    splits = get_chain_splits(chain)
    fracs = sorted(splits.keys())
    result = {}
    for i in range(len(fracs) - 1):
        prev_f, next_f = fracs[i], fracs[i + 1]
        prev_set = parse_split(splits[prev_f])
        next_set = parse_split(splits[next_f])
        dropped = prev_set - next_set
        added = next_set - prev_set
        result[f"{prev_f}→{next_f}"] = {
            "monotonic": len(dropped) == 0,
            "dropped": len(dropped),
            "added": len(added),
            "prev_size": len(prev_set),
            "next_size": len(next_set),
        }
    return result


# ---------------------------------------------------------------------------
# Aggregate analysis helpers
# ---------------------------------------------------------------------------

def class_dist_for_chain(chain: dict) -> dict[str, Counter]:
    """For each step in a chain, return the class distribution of ADDED images."""
    additions = get_chain_additions(chain)
    return {step: class_distribution(stems) for step, stems in additions.items()}


def full_pool_class_distribution(labels_dir: Path = LABELS_DIR) -> Counter:
    """Class distribution of the entire training pool."""
    all_stems = {p.stem for p in labels_dir.glob("*.txt")}
    return class_distribution(all_stems)
