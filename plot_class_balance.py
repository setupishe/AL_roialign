import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NC = len(VOC_CLASSES)

LABELS_DIR = '/home/setupishe/datasets/VOC/labels/train'
VOC_DIR = '/home/setupishe/datasets/VOC'
FRACTIONS = [0.3, 0.4, 0.5, 0.6, 0.7]

STRATEGIES = {
    'distance': 'train_{f}_distance.txt',
    'matr_everything_m': 'train_{f}_distance_matryoshka_everything_really_everything_m.txt',
}


def load_image_list(txt_path):
    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    # extract stem: ./images/train/000966.jpg -> 000966
    stems = set()
    for l in lines:
        stem = os.path.splitext(os.path.basename(l))[0]
        stems.add(stem)
    return stems


def count_boxes(stems):
    counts = np.zeros(NC, dtype=int)
    for stem in stems:
        lbl = os.path.join(LABELS_DIR, stem + '.txt')
        if not os.path.exists(lbl):
            continue
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if line:
                    cls = int(line.split()[0])
                    if 0 <= cls < NC:
                        counts[cls] += 1
    return counts


# Load cumulative sets per strategy per fraction
data = {}
for strat, tmpl in STRATEGIES.items():
    data[strat] = {}
    for f in FRACTIONS:
        fname = tmpl.format(f=f)
        path = os.path.join(VOC_DIR, fname)
        if os.path.exists(path):
            data[strat][f] = load_image_list(path)
        else:
            print(f"MISSING: {path}")

# Compute delta (newly added) box counts per step
delta = {}
for strat in STRATEGIES:
    delta[strat] = {}
    prev = set()
    for f in FRACTIONS:
        if f not in data[strat]:
            continue
        curr = data[strat][f]
        new_imgs = curr - prev
        delta[strat][f] = count_boxes(new_imgs)
        prev = curr

# Also compute cumulative box counts
cumul = {}
for strat in STRATEGIES:
    cumul[strat] = {}
    for f in FRACTIONS:
        if f not in data[strat]:
            continue
        cumul[strat][f] = count_boxes(data[strat][f])

# ---- Plot ----
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Class Balance: Distance vs Matryoshka-Everything (m model)\nVOC 20–70% regime', fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(2, len(FRACTIONS), hspace=0.45, wspace=0.3)

strat_labels = {'distance': 'Distance', 'matr_everything_m': 'Matr-Everything (m)'}
colors = {'distance': '#4C72B0', 'matr_everything_m': '#DD8452'}

# Compute full dataset balance
print("Computing full dataset balance...")
full_stems = load_image_list(os.path.join(VOC_DIR, 'train.txt'))
full_counts = count_boxes(full_stems)
full_norm = full_counts / full_counts.sum() * 100

x = np.arange(NC)
bar_w = 0.28

# Pre-compute shared y limits per row
all_cum_max, all_del_max = [], []
for f in FRACTIONS:
    for strat in STRATEGIES:
        if f in cumul[strat]:
            norm = cumul[strat][f] / cumul[strat][f].sum() * 100
            all_cum_max.append(norm.max())
        if f in delta[strat]:
            total = delta[strat][f].sum()
            if total > 0:
                all_del_max.append((delta[strat][f] / total * 100).max())
all_cum_max.append(full_norm.max())
all_del_max.append(full_norm.max())
ylim_cum = max(all_cum_max) * 1.15
ylim_del = max(all_del_max) * 1.15

for col, f in enumerate(FRACTIONS):
    # --- Top row: cumulative ---
    ax_cum = fig.add_subplot(gs[0, col])
    # full dataset as step line behind bars
    ax_cum.step(x, full_norm, where='mid', color='green', linewidth=1.5, alpha=0.8,
                label='Full dataset', linestyle='--')
    for i, (strat, color) in enumerate(colors.items()):
        if f in cumul[strat]:
            counts = cumul[strat][f]
            norm = counts / counts.sum() * 100
            offset = (i - 0.5) * bar_w
            ax_cum.bar(x + offset, norm, bar_w, color=color, alpha=0.85, label=strat_labels[strat])
    ax_cum.set_title(f'f={f}', fontsize=10)
    ax_cum.set_xticks(x)
    ax_cum.set_xticklabels(VOC_CLASSES, rotation=90, fontsize=6)
    ax_cum.set_ylim(0, ylim_cum)
    ax_cum.set_ylabel('% of boxes' if col == 0 else '')
    if col == 0:
        ax_cum.legend(fontsize=7, loc='upper right')
    ax_cum.axvline(x=14, color='red', alpha=0.3, linewidth=1, linestyle='--')

    # --- Bottom row: delta (newly added) ---
    ax_del = fig.add_subplot(gs[1, col])
    ax_del.step(x, full_norm, where='mid', color='green', linewidth=1.5, alpha=0.8,
                label='Full dataset', linestyle='--')
    for i, (strat, color) in enumerate(colors.items()):
        if f in delta[strat]:
            counts = delta[strat][f]
            total = counts.sum()
            norm = counts / total * 100 if total > 0 else counts
            offset = (i - 0.5) * bar_w
            ax_del.bar(x + offset, norm, bar_w, color=color, alpha=0.85, label=strat_labels[strat])
    ax_del.set_title(f'Δ at f={f}', fontsize=10)
    ax_del.set_xticks(x)
    ax_del.set_xticklabels(VOC_CLASSES, rotation=90, fontsize=6)
    ax_del.set_ylim(0, ylim_del)
    ax_del.set_ylabel('% of new boxes' if col == 0 else '')
    ax_del.axvline(x=14, color='red', alpha=0.3, linewidth=1, linestyle='--')

# Add row labels
fig.text(0.01, 0.72, 'CUMULATIVE', va='center', rotation='vertical', fontsize=11, fontweight='bold')
fig.text(0.01, 0.27, 'NEWLY ADDED', va='center', rotation='vertical', fontsize=11, fontweight='bold')

out = '/home/setupishe/bel_conf/class_balance_dist_vs_matr.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()
