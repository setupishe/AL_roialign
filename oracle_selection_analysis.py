"""
Oracle Selection Analysis for Matryoshka AL
Build RGC using GT, compare actual AL selections against it.
"""
import os, glob, sys
import numpy as np
from collections import Counter
from tqdm import tqdm

DATASETS_DIR = "/home/setupishe/datasets"
DATASET = "VOC"
NUM_CLASSES = 20
LABELS_DIR = f"{DATASETS_DIR}/{DATASET}/labels/train"
IMAGES_DIR = f"{DATASETS_DIR}/{DATASET}/images/train"


def load_train_list(path):
    with open(path) as f:
        return [os.path.basename(l.strip()) for l in f if l.strip()]


def load_gt_labels(img_name):
    label_file = os.path.join(LABELS_DIR, img_name.replace(".jpg", ".txt"))
    if not os.path.exists(label_file) or os.path.getsize(label_file) == 0:
        return []
    with open(label_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    boxes = []
    for line in lines:
        parts = line.split()
        cls = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append({"cls": cls, "xc": xc, "yc": yc, "w": w, "h": h, "area": w * h})
    return boxes


def build_gt_database():
    all_images = sorted(glob.glob(f"{IMAGES_DIR}/*.jpg"))
    db = {}
    for img_path in tqdm(all_images, desc="Loading GT"):
        name = os.path.basename(img_path)
        db[name] = load_gt_labels(name)
    return db


def compute_pool_stats(pool_names, gt_db):
    class_counts = Counter()
    total_boxes = 0
    area_sum = 0.0
    for name in pool_names:
        for box in gt_db.get(name, []):
            class_counts[box["cls"]] += 1
            total_boxes += 1
            area_sum += box["area"]
    return {
        "class_counts": class_counts,
        "total_boxes": total_boxes,
        "classes_covered": len(class_counts),
        "avg_area": area_sum / max(1, total_boxes),
    }


def score_image_for_pool(img_name, gt_db, pool_stats):
    boxes = gt_db.get(img_name, [])
    if not boxes:
        return 0.0
    cc = pool_stats["class_counts"]
    total = pool_stats["total_boxes"]
    rare_score = 0.0
    for box in boxes:
        freq = cc.get(box["cls"], 0) / max(1, total)
        rare_score += 1.0 / (freq + 0.01)
    classes_in_img = set(b["cls"] for b in boxes)
    diversity_score = len(classes_in_img)
    box_count_score = len(boxes)
    areas = [b["area"] for b in boxes]
    avg_area = pool_stats["avg_area"]
    size_diversity = sum(abs(a - avg_area) for a in areas) / max(1, len(areas))
    score = rare_score * 1.0 + diversity_score * 2.0 + box_count_score * 0.5 + size_diversity * 5.0
    return score


def fast_oracle_selection(pool_names, all_names, gt_db, k, batch_size=None):
    pool = set(pool_names)
    remaining = [n for n in all_names if n not in pool]
    selected = []
    if batch_size is None:
        batch_size = max(1, k // 10)
    pool_stats = compute_pool_stats(pool, gt_db)
    while len(selected) < k and remaining:
        scores = [(score_image_for_pool(img, gt_db, pool_stats), img) for img in remaining]
        scores.sort(key=lambda x: x[0], reverse=True)
        batch_pick = min(batch_size, k - len(selected), len(scores))
        for i in range(batch_pick):
            selected.append(scores[i][1])
            pool.add(scores[i][1])
        remaining = [img for img in remaining if img not in pool]
        pool_stats = compute_pool_stats(pool, gt_db)
    return selected[:k]


def jaccard(set1, set2):
    s1, s2 = set(set1), set(set2)
    return len(s1 & s2) / max(1, len(s1 | s2))


def analyze_selection_quality(selected_names, gt_db):
    class_counts = Counter()
    total_boxes = 0
    areas = []
    empty = 0
    for name in selected_names:
        boxes = gt_db.get(name, [])
        if not boxes:
            empty += 1
            continue
        for box in boxes:
            class_counts[box["cls"]] += 1
            total_boxes += 1
            areas.append(box["area"])
    return {
        "n_images": len(selected_names),
        "n_empty": empty,
        "total_boxes": total_boxes,
        "boxes_per_image": total_boxes / max(1, len(selected_names)),
        "classes_covered": len(class_counts),
        "class_distribution": dict(class_counts),
        "avg_area": float(np.mean(areas)) if areas else 0,
    }


def main():
    print("=" * 60)
    print("ORACLE SELECTION ANALYSIS")
    print("=" * 60)

    gt_db = build_gt_database()
    all_names = sorted(gt_db.keys())
    print(f"\nLoaded GT for {len(gt_db)} images")
    all_boxes = sum(len(v) for v in gt_db.values())
    empty_count = sum(1 for v in gt_db.values() if not v)
    print(f"Total boxes: {all_boxes}, Empty images: {empty_count}")

    step_pairs = [
        (0.05, 0.075), (0.075, 0.1), (0.1, 0.125),
        (0.125, 0.15), (0.15, 0.175), (0.175, 0.2),
    ]

    for from_f, to_f in step_pairs:
        print(f"\n{'='*50}")
        print(f"  STEP {from_f} -> {to_f}")
        print(f"{'='*50}")

        if from_f == 0.05:
            base_from = matr_from = f"{DATASETS_DIR}/{DATASET}/train_{from_f}.txt"
        else:
            base_from = f"{DATASETS_DIR}/{DATASET}/train_{from_f}_distance_s_5_20_night_pseudo.txt"
            matr_from = f"{DATASETS_DIR}/{DATASET}/train_{from_f}_distance_matryoshka_everything_really_everything_s_5_20_pseudo.txt"

        base_to = f"{DATASETS_DIR}/{DATASET}/train_{to_f}_distance_s_5_20_night_pseudo.txt"
        matr_to = f"{DATASETS_DIR}/{DATASET}/train_{to_f}_distance_matryoshka_everything_really_everything_s_5_20_pseudo.txt"

        for p in [base_from, matr_from, base_to, matr_to]:
            if not os.path.exists(p):
                print(f"  SKIP: {os.path.basename(p)} missing")
                continue

        base_from_names = set(load_train_list(base_from))
        matr_from_names = set(load_train_list(matr_from))
        base_to_names = load_train_list(base_to)
        matr_to_names = load_train_list(matr_to)

        base_added = [n for n in base_to_names if n not in base_from_names]
        matr_added = [n for n in matr_to_names if n not in matr_from_names]
        k = min(len(base_added), len(matr_added))

        print(f"  Base added: {len(base_added)}, Matr added: {len(matr_added)}")
        print(f"  Selection overlap (Jaccard): {jaccard(base_added, matr_added):.3f}")

        bq = analyze_selection_quality(base_added, gt_db)
        mq = analyze_selection_quality(matr_added, gt_db)

        print(f"\n  {'Metric':<25} {'Baseline':>10} {'Matryoshka':>12}")
        print(f"  {'-'*49}")
        print(f"  {'Boxes/image':<25} {bq['boxes_per_image']:>10.2f} {mq['boxes_per_image']:>12.2f}")
        print(f"  {'Classes covered':<25} {bq['classes_covered']:>10d} {mq['classes_covered']:>12d}")
        print(f"  {'Empty images':<25} {bq['n_empty']:>10d} {mq['n_empty']:>12d}")
        print(f"  {'Total boxes':<25} {bq['total_boxes']:>10d} {mq['total_boxes']:>12d}")
        print(f"  {'Avg area':<25} {bq['avg_area']:>10.4f} {mq['avg_area']:>12.4f}")

        # Big class diffs
        all_cls = sorted(set(list(bq["class_distribution"].keys()) + list(mq["class_distribution"].keys())))
        big_diffs = []
        for c in all_cls:
            bc = bq["class_distribution"].get(c, 0)
            mc = mq["class_distribution"].get(c, 0)
            diff = bc - mc
            if abs(diff) > 3:
                big_diffs.append((c, bc, mc, diff))
        if big_diffs:
            print(f"\n  Class diffs (|base-matr|>3):")
            for c, bc, mc, diff in big_diffs:
                print(f"    Cls {c:2d}: base={bc:4d} matr={mc:4d} diff={diff:+4d}")

        # Oracle
        pool_for_oracle = list(base_from_names)
        print(f"\n  Computing oracle (k={k})...")
        oracle_sel = fast_oracle_selection(pool_for_oracle, all_names, gt_db, k)
        oq = analyze_selection_quality(oracle_sel, gt_db)

        print(f"  Oracle: boxes/img={oq['boxes_per_image']:.2f}, classes={oq['classes_covered']}, empty={oq['n_empty']}")

        base_j = jaccard(base_added, oracle_sel)
        matr_j = jaccard(matr_added, oracle_sel)
        print(f"\n  Overlap with Oracle:")
        print(f"    Baseline:   {base_j:.3f}")
        print(f"    Matryoshka: {matr_j:.3f}")
        winner = "BASE" if base_j > matr_j else "MATR" if matr_j > base_j else "TIE"
        print(f"    >>> {winner} closer to oracle")


if __name__ == "__main__":
    main()
