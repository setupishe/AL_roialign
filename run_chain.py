#!/usr/bin/env python3
"""Active Learning Chain Runner — reads YAML configs and orchestrates the pipeline.

Usage:
    python3 run_chain.py configs/distance_matryoshka_nested.yaml
    python3 run_chain.py configs/random_matryoshka_weights_decreasing.yaml
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


# ── git change detection ──────────────────────────────────────────────────────

def check_uncommitted_py_files() -> None:
    """Warn if any Python files in the repo have uncommitted changes."""
    try:
        modified = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()

        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()

        changed_py = sorted({f for f in modified + untracked if f.endswith(".py")})
    except (subprocess.CalledProcessError, FileNotFoundError):
        return  # not a git repo or git not found — skip

    if not changed_py:
        return

    print("WARNING: Uncommitted changes detected in Python files:")
    for f in changed_py:
        print(f"  {f}")
    answer = input("Proceed anyway? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(1)


# ── template helpers ──────────────────────────────────────────────────────────

def expand(value: object, ctx: dict) -> str:
    """Expand ${key} placeholders in *value* (converted to str first)."""
    s = str(value)
    for k, v in ctx.items():
        s = s.replace(f"${{{k}}}", str(v))
    return s


def build_yolo_args(template: dict, ctx: dict) -> list[str]:
    """Return ['key=value', ...] list suitable for `yolo train`, with template expansion."""
    return [f"{k}={expand(v, ctx)}" for k, v in template.items()]


# ── experiment dir helpers ────────────────────────────────────────────────────

def _yolo_save_dir_candidates(
    run_name: str,
    yolo_template: dict | None,
    ctx: dict | None,
) -> list[Path]:
    """Paths where Ultralytics may have written the run (CWD vs global runs_dir vs explicit project)."""
    ctx = ctx or {}
    task = str((yolo_template or {}).get("task", "detect"))
    out: list[Path] = []

    if yolo_template and (raw := yolo_template.get("project")):
        out.append(Path(expand(raw, ctx)) / run_name)

    try:
        from ultralytics.utils import RUNS_DIR

        out.append(RUNS_DIR / task / run_name)
    except ImportError:
        out.append(Path.home() / "ultralytics" / "runs" / task / run_name)

    out.extend(
        [
            Path("runs") / task / run_name,
            Path("runs") / "detect" / run_name,
            Path("runs") / run_name,
        ]
    )
    # De-dupe while preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def save_config_to_exp(
    config_path: str,
    run_name: str,
    yolo_template: dict | None = None,
    ctx: dict | None = None,
) -> None:
    """Copy the YAML config into the YOLO experiment output directory."""
    for candidate in _yolo_save_dir_candidates(run_name, yolo_template, ctx):
        if candidate.exists():
            dest = candidate / Path(config_path).name
            shutil.copy2(config_path, dest)
            print(f"  Config saved → {dest}")
            return
    print(f"  (Run dir for '{run_name}' not found — config not saved there)")


# ── active-learning chain ─────────────────────────────────────────────────────

def run_active_learning(cfg: dict, config_path: str) -> None:
    ranges: list[float]     = cfg["ranges"]
    mode: str               = cfg["mode"]
    dataset_name: str       = cfg["dataset_name"]
    device                  = cfg.get("device", 0)
    split_name: str         = cfg.get("split_name", mode)
    bg2all_ratio            = cfg.get("bg2all_ratio", 0)
    prepare_script: str     = cfg.get("prepare_script", "prepare_al_split.py")
    weights_base_path: str  = cfg.get("weights_base_path", "runs/detect")
    weights_template: str | None = cfg.get("weights_template")
    # fromsplit_suffix: what to append to "train_{range}" for the from-split file
    # on the first range it is always "" (= random baseline)
    fromsplit_suffix: str   = cfg.get("fromsplit_suffix", f"_{mode}")
    tune: bool              = cfg.get("tune", False)
    prepare_args: dict      = cfg.get("prepare_args", {})
    yolo_template: dict     = cfg.get("yolo_args", {})

    print("=== Active Learning Chain Runner ===")
    print(f"Config:  {config_path}")
    print(f"Mode:    {mode}  |  Split: {split_name}")
    print(f"Dataset: {dataset_name}  |  Device: {device}")
    print(f"Ranges:  {ranges}  |  Tune: {tune}")
    print("====================================\n")

    for range_val in ranges:
        next_range    = round(range_val + 0.1, 1)
        range_str     = f"{range_val}"
        next_range_str = f"{next_range:.1f}"
        is_first      = (range_val == ranges[0])
        folder_name   = "random" if is_first else mode
        from_suffix   = "" if is_first else fromsplit_suffix

        ctx = {
            "mode":             mode,
            "DATASET_NAME":     dataset_name,
            "dataset_name":     dataset_name,
            "DEVICE":           str(device),
            "device":           str(device),
            "next_range":       next_range_str,
            "split_name":       split_name,
            "SPLIT_NAME":       split_name,
            "range":            range_str,
            "folder_name":      folder_name,
            "weights_base_path": weights_base_path,
        }

        # ── weights path ──────────────────────────────────────────────────────
        if weights_template:
            weights_path = expand(weights_template, ctx)
        else:
            weights_path = f"{weights_base_path}/VOC_{folder_name}_{range_val}/weights/best.pt"

        from_split = f"train_{range_val}{from_suffix}.txt"

        print(f"── PREPARE  {range_val} → {next_range_str} ──────────────────")

        # ── build prepare command ─────────────────────────────────────────────
        if prepare_script == "conf_criteria.py":
            cmd = [
                "python3", "conf_criteria.py",
                "--weights", weights_path,
                "--from-fraction", range_str,
                "--to-fraction", next_range_str,
                "--from-split", from_split,
                "--dataset-name", dataset_name,
                "--default-split", "train.txt",
                "--split-name", split_name,
                "--bg2all-ratio", str(bg2all_ratio),
            ]
            if prepare_args.get("cleanup"):
                cmd.append("--cleanup")
        else:
            cmd = [
                "python3", prepare_script,
                "--weights", weights_path,
                "--from-fraction", range_str,
                "--to-fraction", next_range_str,
                "--from-split", from_split,
                "--dataset-name", dataset_name,
                "--split-name", split_name,
                "--mode", mode,
                "--bg2all-ratio", str(bg2all_ratio),
                "--device", str(device),
            ]
            if prepare_args.get("seg2line"):
                cmd.append("--seg2line")
            if prepare_args.get("cleanup"):
                cmd.append("--cleanup")
            if prepare_args.get("skip_pca"):
                cmd.append("--skip-pca")
            if pa := prepare_args.get("index_backend"):
                cmd.extend(["--index-backend", pa])
            if prepare_args.get("coarse_to_fine"):
                cmd.append("--coarse-to-fine")
            if hw := prepare_args.get("roi_hw"):
                cmd.extend(["--roi-hw"] + [str(x) for x in hw])
            if nl := prepare_args.get("netron_layer_names"):
                cmd.extend(["--netron-layer-names", nl])

        subprocess.run(cmd, check=True)

        # ── build YOLO args ───────────────────────────────────────────────────
        yolo_args = build_yolo_args(yolo_template, ctx)

        if tune:
            base_epochs       = int(yolo_template.get("epochs", 30))
            base_lr0          = float(yolo_template.get("lr0", 0.01))
            base_close_mosaic = int(yolo_template.get("close_mosaic", 10))
            rel_increase = (next_range - range_val) / range_val if range_val != 0 else 0
            new_epochs       = max(1, round(base_epochs * rel_increase))
            new_close_mosaic = max(1, round(base_close_mosaic * rel_increase))
            new_lr0          = 0.000004
            print(f"  Tune: epochs={new_epochs}, lr0={new_lr0}, close_mosaic={new_close_mosaic}")
            yolo_args = [
                a for a in yolo_args
                if not a.startswith(("epochs=", "lr0=", "close_mosaic=", "optimizer="))
            ]
            yolo_args += [
                f"epochs={new_epochs}",
                f"lr0={new_lr0}",
                f"close_mosaic={new_close_mosaic}",
                "optimizer=AdamW",
            ]

        # ── confirm & train ───────────────────────────────────────────────────
        print(f"\n── TRAIN  {next_range_str} ──────────────────────────────────")
        answer = input(f"Proceed with training for {next_range_str}? [y/N] ").strip().lower()
        if answer != "y":
            print("Training cancelled. Exiting.")
            sys.exit(0)

        subprocess.run(["yolo", "train"] + yolo_args, check=True)

        run_name = next(
            (a.split("=", 1)[1] for a in yolo_args if a.startswith("name=")),
            None,
        )
        if run_name:
            save_config_to_exp(config_path, run_name, yolo_template, ctx)

        print(f"  Done: {next_range_str}\n")

    print("=== Active Learning Chain Complete ===")


# ── random (sequential) training ─────────────────────────────────────────────

def run_random_train(cfg: dict, config_path: str) -> None:
    fractions: list  = cfg.get("fractions", [])
    yolo_template: dict = cfg.get("yolo_args", {})

    print("=== Random Training Runner ===")
    print(f"Config:    {config_path}")
    print(f"Fractions: {fractions}")
    print("==============================\n")

    for frac in fractions:
        frac_str = str(frac)
        ctx = {"fraction": frac_str}
        yolo_args = build_yolo_args(yolo_template, ctx)

        print(f"── TRAIN  fraction={frac_str} ──────────────────────────")
        subprocess.run(["yolo", "train"] + yolo_args, check=True)

        run_name = next(
            (a.split("=", 1)[1] for a in yolo_args if a.startswith("name=")),
            None,
        )
        if run_name:
            save_config_to_exp(config_path, run_name, yolo_template, ctx)

    print("\n=== Training Complete ===")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Active Learning Chain Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_chain.py configs/distance_matryoshka_nested.yaml
  python3 run_chain.py configs/distance.yaml
  python3 run_chain.py configs/random_matryoshka_weights_decreasing.yaml
""",
    )
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    check_uncommitted_py_files()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_type = cfg.get("type", "active_learning")
    if run_type == "active_learning":
        run_active_learning(cfg, config_path)
    elif run_type == "random_train":
        run_random_train(cfg, config_path)
    else:
        print(f"Error: unknown run type '{run_type}' (expected active_learning or random_train)")
        sys.exit(1)


if __name__ == "__main__":
    main()
