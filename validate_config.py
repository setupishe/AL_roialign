#!/usr/bin/env python3
"""
Config sanity checker for run_chain.py YAML configs.

Catches the class of mistakes where flags are copy-pasted from unrelated configs
and silently corrupt experimental conditions (e.g., seg2line=true on VOC bbox data).

Usage:
    python validate_config.py configs/my_config.yaml          # check one
    python validate_config.py configs/                        # check all in dir
    python validate_config.py --all                           # check all in configs/

Exit code 1 if any FAIL found.
"""

import sys
import yaml
from pathlib import Path

ERRORS = []
WARNINGS = []


def err(msg):
    ERRORS.append(msg)


def warn(msg):
    WARNINGS.append(msg)


# ---------------------------------------------------------------------------
# Dataset knowledge
# ---------------------------------------------------------------------------

# Datasets whose labels are YOLO bbox format (class cx cy w h).
# seg2line=true on these corrupts ROI bboxes used for embedding extraction.
BBOX_ONLY_DATASETS = {"VOC", "voc", "COCO", "coco"}

# MatrE markers — presence in yolo_args or prepare_args identifies a MatrE config.
MATRE_YOLO_MARKERS = {"matryoshka", "matryoshka_shared_assign"}
MATRE_PREPARE_MARKER = "skip_pca"

# Keys that must NOT appear in plain distance configs (belong to MatrE only).
MATRE_ONLY_PREPARE_KEYS = {"skip_pca", "coarse_to_fine", "use_dim"}


def is_matre(cfg: dict) -> bool:
    yolo = cfg.get("yolo_args", {})
    prepare = cfg.get("prepare_args", {})
    return bool(
        set(yolo.keys()) & MATRE_YOLO_MARKERS
        or prepare.get(MATRE_PREPARE_MARKER)
    )


def is_random_train(cfg: dict) -> bool:
    return cfg.get("type") == "random_train"


def _parse_matryoshka_list(s: str) -> list:
    """Parse '[0.2,0.4,1.0]' or '8,4,1' into a list."""
    return [x.strip() for x in s.strip("[]").split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def check_matryoshka_weights(yolo: dict) -> None:
    """RULE 9: matryoshka_weights length must match matryoshka_granularity_divs length."""
    divs_raw = yolo.get("matryoshka_granularity_divs")
    weights_raw = yolo.get("matryoshka_weights")
    if divs_raw is None or weights_raw is None:
        return
    divs = _parse_matryoshka_list(str(divs_raw))
    weights = _parse_matryoshka_list(str(weights_raw))
    if len(divs) != len(weights):
        err(
            f"[matr_weights_len] matryoshka_weights has {len(weights)} elements {weights} "
            f"but matryoshka_granularity_divs has {len(divs)} levels {divs}. "
            f"Lengths must match."
        )


def check(cfg: dict, path: Path) -> None:
    if is_random_train(cfg):
        # random_train configs are a different format — only check seed + matryoshka
        yolo = cfg.get("yolo_args", {})
        if yolo.get("seed") is None:
            warn("[no_seed] yolo_args.seed absent — results not reproducible.")
        check_matryoshka_weights(yolo)
        return

    dataset = cfg.get("dataset_name", "")
    prepare = cfg.get("prepare_args", {})
    yolo = cfg.get("yolo_args", {})
    ranges = cfg.get("ranges", [])
    mode = cfg.get("mode", "")
    matre = is_matre(cfg)

    # ------------------------------------------------------------------
    # RULE 1 (CRITICAL): seg2line must be false/absent for bbox datasets
    # seg2line treats (cx cy w h) as a 2-point polygon → corrupted bbox
    # Example: cx=0.5 cy=0.5 w=0.3 h=0.4 → output cx=0.4 cy=0.45 w=0.2 h=0.1
    # This corrupts ALL ROI coordinates used for embedding extraction.
    # ------------------------------------------------------------------
    if prepare.get("seg2line") and dataset in BBOX_ONLY_DATASETS:
        err(
            f"[seg2line] seg2line=true on '{dataset}' (YOLO bbox format). "
            f"Treats (cx cy w h) as a polygon → corrupted ROI bboxes for embeddings. "
            f"seg2line is for segmentation datasets only (e.g., COCO-seg). Remove it."
        )

    # ------------------------------------------------------------------
    # RULE 2: ranges must have at least 2 values for AL chain configs
    # ------------------------------------------------------------------
    if mode in ("distance", "density", "confidence") and len(ranges) < 2:
        err(
            f"[ranges] mode='{mode}' requires at least 2 values in ranges "
            f"(from_fraction + one or more to_fractions). Got: {ranges}"
        )

    # ------------------------------------------------------------------
    # RULE 3: MatrE-only keys must not appear in non-MatrE distance configs
    # ------------------------------------------------------------------
    if mode == "distance" and not matre:
        for key in MATRE_ONLY_PREPARE_KEYS:
            if prepare.get(key):
                err(
                    f"[matre_leak] prepare_args.{key}=true in a plain distance config. "
                    f"This key belongs to MatrE configs only — likely copied from a MatrE config."
                )

    # ------------------------------------------------------------------
    # RULE 4: skip_pca + use_standard_scaler are mutually exclusive
    # ------------------------------------------------------------------
    if prepare.get("skip_pca") and prepare.get("use_standard_scaler"):
        err(
            f"[pca_conflict] skip_pca=true and use_standard_scaler=true are mutually exclusive."
        )

    # ------------------------------------------------------------------
    # RULE 5: pretrained must be explicit in scratch experiments
    # ------------------------------------------------------------------
    pretrained = yolo.get("pretrained")
    if pretrained is None and "scratch" in str(path):
        warn(
            f"[pretrained] Config path contains 'scratch' but yolo_args.pretrained is absent. "
            f"Omitting it defaults to pretrained weights and silently invalidates scratch runs."
        )

    # ------------------------------------------------------------------
    # RULE 6: seed must be set
    # ------------------------------------------------------------------
    if yolo.get("seed") is None:
        warn("[no_seed] yolo_args.seed absent — results not reproducible.")

    # ------------------------------------------------------------------
    # RULE 7: distance configs without netron_layer_names
    # ------------------------------------------------------------------
    if mode == "distance" and not prepare.get("netron_layer_names") and not matre:
        warn(
            f"[no_layers] mode=distance, no netron_layer_names — embeddings from default layers. "
            f"Confirm this is intended."
        )

    # ------------------------------------------------------------------
    # RULE 9: matryoshka_weights length must match granularity_divs length
    # ------------------------------------------------------------------
    check_matryoshka_weights(yolo)

    # ------------------------------------------------------------------
    # RULE 8: COCO without train2017 subdir
    # ------------------------------------------------------------------
    if dataset in {"COCO", "coco"}:
        if prepare.get("train_subdir", "train") == "train":
            warn(
                f"[coco_subdir] dataset=COCO but train_subdir='train'. "
                f"COCO images live under train2017/. Set train_subdir: train2017."
            )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def check_file(path: Path) -> bool:
    global ERRORS, WARNINGS
    ERRORS = []
    WARNINGS = []

    try:
        cfg = yaml.safe_load(path.read_text())
    except Exception as e:
        print(f"[PARSE ERROR] {path}: {e}")
        return False

    if not isinstance(cfg, dict):
        print(f"[SKIP] {path}: not a YAML dict")
        return True

    check(cfg, path)

    ok = not bool(ERRORS)
    if ERRORS or WARNINGS:
        print(f"\n{'='*60}")
        print(f"Config: {path}")
        for e in ERRORS:
            print(f"  FAIL  {e}")
        for w in WARNINGS:
            print(f"  WARN  {w}")
    else:
        print(f"OK    {path}")

    return ok


def main():
    args = sys.argv[1:]
    if not args or args == ["--all"]:
        targets = sorted(Path("configs").glob("*.yaml"))
    else:
        targets = []
        for a in args:
            p = Path(a)
            if p.is_dir():
                targets.extend(sorted(p.glob("*.yaml")))
            else:
                targets.append(p)

    if not targets:
        print("No YAML configs found.")
        sys.exit(0)

    all_ok = True
    for t in targets:
        if not check_file(t):
            all_ok = False

    print()
    if all_ok:
        print(f"All {len(targets)} configs passed (no FAIL).")
        sys.exit(0)
    else:
        print(f"FAILURES detected. Fix before running.")
        sys.exit(1)


if __name__ == "__main__":
    main()
