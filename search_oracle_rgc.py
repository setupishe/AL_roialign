"""RGC Oracle Image Selection — VOC / COCO active learning.

Candidates:
  miss_score   ChatGPT model-failure-aware oracle. Requires --weights.
               Score = sum_c  w_c * log(1 + sum_{j in class c} m_j)
               where w_c = 1/sqrt(N_c_labeled + 1) and
               m_j = fraction of IoU thresholds {0.50..0.95} that the best
               matching prediction misses for GT object j.
  balanced_v1  Simple GT-only heuristic (kept for comparison).

Usage:
  # Dry run — print stats for both candidates
  python search_oracle_rgc.py \\
    --weights /path/to/VOC_random_0.2_s/weights/best.pt \\
    --candidates miss_score balanced_v1

  # Write winning split + dataset YAML
  python search_oracle_rgc.py \\
    --weights /path/to/best.pt \\
    --write-candidate miss_score \\
    --output-split /path/to/datasets/VOC/train_0.3_oracle_miss_score_s.txt \\
    --template-yaml /path/to/ultralytics/cfg/datasets/VOC.yaml \\
    --output-yaml /path/to/ultralytics/cfg/datasets/VOC_0.3_oracle_miss_score_s.yaml
"""
import argparse
import math
import os
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# IoU thresholds (COCO-aligned, 10 thresholds)
# ---------------------------------------------------------------------------
_IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05)  # [0.50, 0.55, ..., 0.95]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_split(path):
    """Return list of basenames from a split file (one ./images/sub/name.jpg per line)."""
    with open(path) as f:
        return [os.path.basename(line.strip()) for line in f if line.strip()]


def build_gt_database(dataset_dir, train_subdir):
    """Return {image_name: [box_dict, ...]} for all images in the train dir.

    box_dict keys: cls, xc, yc, w, h, area, x1, y1, x2, y2 (all normalized).
    """
    labels_dir = os.path.join(dataset_dir, "labels", train_subdir)
    images_dir = os.path.join(dataset_dir, "images", train_subdir)
    db = {}
    for image_name in sorted(os.listdir(images_dir)):
        if not image_name.lower().endswith(".jpg"):
            continue
        stem = image_name.rsplit(".", 1)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        boxes = []
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(parts[0])
                    xc, yc, w, h = (float(p) for p in parts[1:5])
                    boxes.append({
                        "cls": cls,
                        "xc": xc, "yc": yc, "w": w, "h": h,
                        "area": w * h,
                        # normalized xyxy (for IoU matching)
                        "x1": xc - w / 2,
                        "y1": yc - h / 2,
                        "x2": xc + w / 2,
                        "y2": yc + h / 2,
                    })
        db[image_name] = boxes
    return db


def compute_pool_stats(pool_names, gt_db):
    """Class instance counts in the labeled pool."""
    class_counts = Counter()
    total_boxes = 0
    for name in pool_names:
        for box in gt_db.get(name, []):
            class_counts[box["cls"]] += 1
            total_boxes += 1
    return {"class_counts": class_counts, "total_boxes": total_boxes}


def build_feature_table(unlabeled_imgs, gt_db, pool_stats):
    """Lightweight per-image features used by balanced_v1."""
    return {img: _image_features(gt_db[img], pool_stats) for img in unlabeled_imgs}


def _image_features(boxes, pool_stats):
    if not boxes:
        return {"count": 0.0, "unique": 0.0, "rarity_avg": 0.0}
    class_counts = pool_stats["class_counts"]
    total_boxes = pool_stats["total_boxes"]
    count = len(boxes)
    unique = len(set(b["cls"] for b in boxes))
    rarity_terms = [
        1.0 / (class_counts.get(b["cls"], 0) / max(1, total_boxes) + 0.001)
        for b in boxes
    ]
    return {
        "count": float(count),
        "unique": float(unique),
        "rarity_avg": float(sum(rarity_terms) / count),
    }


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _box_iou_xyxy(a, b):
    """Vectorized IoU between two arrays of normalized xyxy boxes.

    a: (N, 4), b: (M, 4) -> returns (N, M) float array.
    """
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


# ---------------------------------------------------------------------------
# Candidate: miss_score  (ChatGPT oracle formula)
# ---------------------------------------------------------------------------

def score_miss_score(unlabeled_imgs, gt_db, from_names, weights_path,
                     dataset_dir, train_subdir, batch_size=16):
    """Model-failure-aware oracle scoring.

    S(x) = sum_c  w_c * log(1 + sum_{j in class c} m_j)
    w_c   = 1 / sqrt(N_c_labeled + 1)
    m_j   = fraction of IoU thresholds {0.50..0.95} that the best
            matching prediction for GT object j fails to reach.
            (m_j = 1.0 if no prediction of that class exists)
    """
    from ultralytics import YOLO

    # Rarity weights from labeled pool
    pool_stats = compute_pool_stats(from_names, gt_db)
    class_counts = pool_stats["class_counts"]

    all_classes = {b["cls"] for boxes in gt_db.values() for b in boxes}
    w_c = {c: 1.0 / math.sqrt(class_counts.get(c, 0) + 1) for c in all_classes}

    # Run inference on unlabeled pool
    images_dir = os.path.join(dataset_dir, "images", train_subdir)
    model = YOLO(weights_path)

    print(f"  Running inference on {len(unlabeled_imgs)} unlabeled images "
          f"(batch={batch_size})...", flush=True)

    # preds[img] = {cls_id: np.array of shape (M, 4) in normalized xyxy}
    preds = {}
    for i in range(0, len(unlabeled_imgs), batch_size):
        batch_names = unlabeled_imgs[i:i + batch_size]
        batch_paths = [os.path.join(images_dir, n) for n in batch_names]
        results = model.predict(batch_paths, verbose=False)
        for name, result in zip(batch_names, results):
            cls_boxes = {}
            if result.boxes is not None and len(result.boxes):
                xyxyn = result.boxes.xyxyn.cpu().numpy()   # normalized xyxy
                classes = result.boxes.cls.cpu().numpy().astype(int)
                for cls_id, box in zip(classes, xyxyn):
                    cls_boxes.setdefault(cls_id, []).append(box)
            # convert lists to arrays
            preds[name] = {c: np.array(v) for c, v in cls_boxes.items()}
        if (i // batch_size) % 20 == 0:
            print(f"  Inference: {min(i + batch_size, len(unlabeled_imgs))}"
                  f"/{len(unlabeled_imgs)}", flush=True)

    print("  Inference done. Computing scores...", flush=True)

    # Score each image
    scores = {}
    for img in unlabeled_imgs:
        gt_boxes = gt_db.get(img, [])
        img_preds = preds.get(img, {})

        # Group GT by class
        gt_by_class = {}
        for b in gt_boxes:
            gt_by_class.setdefault(b["cls"], []).append(b)

        S = 0.0
        for cls_id, gt_list in gt_by_class.items():
            gt_arr = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]]
                                for b in gt_list])  # (N, 4)

            pred_arr = img_preds.get(cls_id)
            if pred_arr is not None and len(pred_arr):
                iou_mat = _box_iou_xyxy(gt_arr, pred_arr)   # (N, M)
                best_iou = iou_mat.max(axis=1)               # (N,)
            else:
                best_iou = np.zeros(len(gt_list))

            # miss fraction per GT object
            miss_fracs = np.mean(best_iou[:, None] < _IOU_THRESHOLDS[None, :],
                                 axis=1)  # (N,)

            E_xc = math.log(1.0 + float(miss_fracs.sum()))
            S += w_c.get(cls_id, 0.0) * E_xc

        scores[img] = S

    return scores


# ---------------------------------------------------------------------------
# Candidate: balanced_v1  (kept for comparison)
# ---------------------------------------------------------------------------

def score_balanced_v1(features_by_img):
    return {
        img: 0.5 * f["count"] + 0.3 * f["rarity_avg"] + 2.0 * f["unique"]
        for img, f in features_by_img.items()
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def select_top_k(scores, k):
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [img for img, _ in ranked[:k]]


def summarize_selection(selected_imgs, gt_db):
    class_counts = Counter()
    total_boxes = 0
    empty = 0
    unique_per_image = []
    boxes_per_image = []
    for img in selected_imgs:
        boxes = gt_db.get(img, [])
        if not boxes:
            empty += 1
            unique_per_image.append(0.0)
            boxes_per_image.append(0.0)
            continue
        boxes_per_image.append(float(len(boxes)))
        unique_per_image.append(float(len(set(b["cls"] for b in boxes))))
        for b in boxes:
            class_counts[b["cls"]] += 1
            total_boxes += 1
    if class_counts:
        probs = np.array(list(class_counts.values()), dtype=float)
        probs /= probs.sum()
        class_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        top_class_share = float(probs.max())
    else:
        class_entropy = 0.0
        top_class_share = 0.0
    return {
        "images": len(selected_imgs),
        "boxes": total_boxes,
        "bpi": total_boxes / max(1, len(selected_imgs)),
        "classes": len(class_counts),
        "class_entropy": class_entropy,
        "top_class_share": top_class_share,
        "empty_images": empty,
        "avg_unique_per_image": float(np.mean(unique_per_image)) if unique_per_image else 0.0,
        "avg_boxes_per_image": float(np.mean(boxes_per_image)) if boxes_per_image else 0.0,
    }


def print_summary(name, summary, overlap=None):
    overlap_text = f" | overlap_vs_baseline={overlap}" if overlap is not None else ""
    print(
        f"{name:>12} | images={summary['images']:5d} | boxes={summary['boxes']:6d}"
        f" | bpi={summary['bpi']:.2f} | classes={summary['classes']:2d}"
        f" | entropy={summary['class_entropy']:.3f} | top1={summary['top_class_share']:.3f}"
        f" | empty={summary['empty_images']:4d} | uniq/img={summary['avg_unique_per_image']:.2f}"
        f" | boxes/img={summary['avg_boxes_per_image']:.2f}{overlap_text}"
    )


def write_split(output_path, from_names, added_names, train_subdir):
    with open(output_path, "w") as f:
        for name in from_names + added_names:
            f.write(f"./images/{train_subdir}/{name}\n")


def write_yaml_from_template(template_yaml, output_yaml, new_train_name):
    with open(template_yaml) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("train:"):
            # Replace whatever is after "train: " with new_train_name
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{indent}train: {new_train_name}\n"
            break
    with open(output_yaml, "w") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RGC Oracle Image Selection")
    parser.add_argument("--dataset-dir", default="/home/setupishe/datasets/VOC")
    parser.add_argument("--train-subdir", default="train")
    parser.add_argument("--from-split",
                        default="/home/setupishe/datasets/VOC/train_0.2.txt")
    parser.add_argument("--baseline-to-split",
                        default="/home/setupishe/datasets/VOC/train_0.3.txt")
    parser.add_argument("--weights", default=None,
                        help="YOLO checkpoint for miss_score inference")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size for miss_score")
    parser.add_argument("--candidates", nargs="+",
                        default=["miss_score", "balanced_v1"])
    parser.add_argument("--write-candidate", default=None)
    parser.add_argument("--output-split", default=None)
    parser.add_argument("--template-yaml",
                        default="/home/setupishe/ultralytics/ultralytics/cfg/datasets/VOC.yaml")
    parser.add_argument("--output-yaml", default=None)
    args = parser.parse_args()

    gt_db = build_gt_database(args.dataset_dir, args.train_subdir)
    from_names = load_split(args.from_split)
    baseline_names = load_split(args.baseline_to_split)
    from_set = set(from_names)
    baseline_added = [n for n in baseline_names if n not in from_set]
    unlabeled_imgs = sorted(n for n in gt_db if n not in from_set)
    k = len(baseline_added)

    print("=" * 80)
    print("VOC ORACLE RGC SEARCH")
    print("=" * 80)
    print(f"Dataset images  : {len(gt_db)}")
    print(f"From split      : {len(from_names)}")
    print(f"Baseline to split: {len(baseline_names)}")
    print(f"Budget k        : {k}")
    print(f"Unlabeled pool  : {len(unlabeled_imgs)}")

    baseline_summary = summarize_selection(baseline_added, gt_db)
    print_summary("baseline", baseline_summary)

    pool_stats = compute_pool_stats(from_names, gt_db)
    features_by_img = build_feature_table(unlabeled_imgs, gt_db, pool_stats)

    selections = {}
    print("-" * 80)
    for candidate in args.candidates:
        print(f"Scoring: {candidate} ...")
        if candidate == "miss_score":
            if not args.weights:
                raise ValueError("--weights is required for miss_score")
            scores = score_miss_score(
                unlabeled_imgs, gt_db, from_names,
                args.weights, args.dataset_dir, args.train_subdir,
                batch_size=args.batch_size,
            )
        elif candidate == "balanced_v1":
            scores = score_balanced_v1(features_by_img)
        else:
            raise ValueError(f"Unknown candidate: {candidate!r}")

        selected = select_top_k(scores, k)
        selections[candidate] = selected
        summary = summarize_selection(selected, gt_db)
        overlap = len(set(selected) & set(baseline_added))
        print_summary(candidate, summary, overlap=overlap)

    if args.write_candidate:
        if args.write_candidate not in selections:
            raise ValueError(f"{args.write_candidate!r} was not scored")
        if not args.output_split:
            raise ValueError("--output-split required with --write-candidate")
        selected = selections[args.write_candidate]
        write_split(args.output_split, from_names, selected, args.train_subdir)
        print(f"Saved split: {args.output_split}")
        if args.output_yaml:
            write_yaml_from_template(
                args.template_yaml,
                args.output_yaml,
                os.path.basename(args.output_split),
            )
            print(f"Saved yaml: {args.output_yaml}")


if __name__ == "__main__":
    main()
