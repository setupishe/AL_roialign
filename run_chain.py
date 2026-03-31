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


# YAML `prepare_args` keys → extra argv for prepare scripts (flag names = key with `_` → `-`).
_PREPARE_BOOL_KEYS = frozenset(
    {
        "batched_inference",
        "save_crops",
        "seg2line",
        "cleanup",
        "skip_pca",
        "coarse_to_fine",
        "from_predictions",
        "separate_maps_voting",
    }
)
_PREPARE_VALUE_KEYS = frozenset(
    {
        "index_backend",
        "io_workers",
        "onnx_batch_size",
        "granularity_divs",
        "ctf_k1_mult",
        "ctf_k2_mult",
        "ctf_d1_div",
        "ctf_d2_div",
        "netron_layer_names",
        "seed",
    }
)
_PREPARE_KNOWN_KEYS = _PREPARE_BOOL_KEYS | _PREPARE_VALUE_KEYS | frozenset({"roi_hw"})
_CONF_CRITERIA_PREPARE_KEYS = frozenset({"seg2line", "cleanup"})


def _prepare_args_to_argv(prepare_args: dict) -> list[str]:
    """Turn cfg `prepare_args` dict into CLI tokens for prepare_al_split.py (and conf_criteria subset)."""
    argv: list[str] = []
    for key, val in prepare_args.items():
        if val is None:
            continue
        if key not in _PREPARE_KNOWN_KEYS:
            raise ValueError(
                f"Unknown prepare_args key '{key}'. Known: {sorted(_PREPARE_KNOWN_KEYS)}"
            )
        if key in _PREPARE_BOOL_KEYS:
            if key == "batched_inference":
                if val is False:
                    argv.append("--no-batched-inference")
            elif val:
                flag = "--" + key.replace("_", "-")
                argv.append(flag)
        elif key == "roi_hw":
            flag = "--" + key.replace("_", "-")
            argv.extend([flag] + [str(x) for x in val])
        else:
            flag = "--" + key.replace("_", "-")
            argv.extend([flag, str(val)])
    return argv


# ── experiment dir helpers ────────────────────────────────────────────────────

def _yolo_run_save_dir(
    run_name: str,
    yolo_template: dict | None,
    ctx: dict | None,
) -> Path:
    """Where Ultralytics writes the run: project/name if project= is set, else RUNS_DIR/task/name."""
    ctx = ctx or {}
    task = str((yolo_template or {}).get("task", "detect"))
    if yolo_template and (raw := yolo_template.get("project")):
        return Path(expand(raw, ctx)) / run_name
    try:
        from ultralytics.utils import RUNS_DIR

        return RUNS_DIR / task / run_name
    except ImportError:
        return Path("runs") / task / run_name


def save_config_to_exp(
    config_path: str,
    run_name: str,
    yolo_template: dict | None = None,
    ctx: dict | None = None,
) -> None:
    """Copy the YAML config into the YOLO experiment output directory."""
    dest_dir = _yolo_run_save_dir(run_name, yolo_template, ctx)
    if not dest_dir.is_dir():
        print(f"  (Run dir not found: {dest_dir} — config not saved)")
        return
    dest = dest_dir / Path(config_path).name
    shutil.copy2(config_path, dest)
    print(f"  Config saved → {dest}")


# ── active-learning chain ─────────────────────────────────────────────────────

def run_active_learning(cfg: dict, config_path: str) -> None:
    ranges: list[float]     = cfg["ranges"]
    mode: str               = cfg["mode"]
    dataset_name: str       = cfg["dataset_name"]
    device                  = cfg.get("device", 0)
    if isinstance(device, list):
        prepare_device = str(device[0])
        yolo_device    = ",".join(str(d) for d in device)
    elif isinstance(device, str) and "," in device:
        prepare_device = device.split(",")[0].strip()
        yolo_device    = device
    else:
        prepare_device = str(device)
        yolo_device    = str(device)
    split_name: str         = cfg.get("split_name", mode)
    bg2all_ratio            = cfg.get("bg2all_ratio", 0)
    prepare_script: str     = cfg.get("prepare_script", "prepare_al_split.py")
    weights_base_path: str  = cfg.get("weights_base_path", "runs/detect")
    weights_template: str | None = cfg.get("weights_template")
    first_weights_template: str | None = cfg.get("first_weights_template")
    datasets_dir: str       = cfg.get("datasets_dir", "/home/setupishe/datasets")
    ultralytics_cfg_dir: str | None = cfg.get("ultralytics_cfg_dir")
    prepare_args: dict      = cfg.get("prepare_args", {})
    yolo_template: dict     = cfg.get("yolo_args", {})
    if len(ranges) < 2:
        raise ValueError("`ranges` must contain at least two values (from and to).")

    print("=== Active Learning Chain Runner ===")
    print(f"Config:  {config_path}")
    print(f"Mode:    {mode}  |  Split: {split_name}")
    print(f"Dataset: {dataset_name}  |  Device (YOLO): {yolo_device}  |  Device (prepare): {prepare_device}")
    print(f"Ranges:  {ranges}")
    print("====================================\n")

    for i, range_val in enumerate(ranges[:-1]):
        next_range    = ranges[i + 1]
        range_str     = f"{range_val}"
        next_range_str = f"{next_range}"
        is_first      = (i == 0)
        folder_name   = "random" if is_first else mode

        ctx = {
            "mode":              mode,
            "DATASET_NAME":      dataset_name,
            "device":            yolo_device,
            "next_range":        next_range_str,
            "split_name":        split_name,
            "range":             range_str,
            "folder_name":       folder_name,
            "weights_base_path": weights_base_path,
        }

        # ── weights path ──────────────────────────────────────────────────────
        active_template = (first_weights_template if is_first and first_weights_template else weights_template)
        if active_template:
            weights_path = expand(active_template, ctx)
        else:
            weights_path = f"{weights_base_path}/{dataset_name}_{folder_name}_{range_val}/weights/best.pt"

        split_root = Path(datasets_dir) / dataset_name
        from_split = (
            f"train_{range_val}.txt"
            if is_first
            else f"train_{range_val}_{split_name}.txt"
        )
        if not (split_root / from_split).is_file():
            raise FileNotFoundError(
                f"from-split not found (expected output of the previous step): {split_root / from_split}"
            )

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
            pa = {k: v for k, v in prepare_args.items() if k in _CONF_CRITERIA_PREPARE_KEYS}
            cmd.extend(_prepare_args_to_argv(pa))
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
                "--device", prepare_device,
                "--datasets-dir", datasets_dir,
            ]
            if ultralytics_cfg_dir:
                cmd.extend(["--ultralytics-cfg-dir", ultralytics_cfg_dir])
            cmd.extend(_prepare_args_to_argv(prepare_args))

        subprocess.run(cmd, check=True)

        # ── build YOLO args ───────────────────────────────────────────────────
        yolo_args = build_yolo_args(yolo_template, ctx)

        # ── train ───────────────────────────────────────────────────────────
        print(f"\n── TRAIN  {next_range_str} ──────────────────────────────────")

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
