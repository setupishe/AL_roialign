#!/usr/bin/env python3
"""
Compare an arbitrary number of training chains (fraction sweeps) from Ultralytics results.csv.

Logic aligned with show_metrics.ipynb: max metric over epochs (fscore = max of 2PR/(P+R) per row).

Plot rule: fraction 0.2 appears only for chains with kind: random. AL chains default to 0.3–0.7.

Usage:
  compare_chains.yaml
  python3 compare_chains.py compare_chains.yaml -o out.png

YAML example:
  runs_dir: /home/setupishe/ultralytics/runs/detect
  metric: metrics/mAP50-95(B)
  title: VOC yolov8s
  output: chain_compare.png
  chains:
    - label: random base
      kind: random
      template: VOC_random_{frac}_s
    - label: random mat
      kind: random
      template: VOC_random_{frac}_s_matryoshka_everything
    - label: distance AL
      kind: al
      template: VOC_distance_{frac}_s_matryoshka_everything_really_everything
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# matplotlib.pyplot imported inside plotting helpers so `main()` can set Agg before first import

DEFAULT_RANDOM_FRACS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_AL_FRACS = (0.3, 0.4, 0.5, 0.6, 0.7)


def normalize_color(value: Any):
    """
    Accept:
      - matplotlib color names: "red", "tab:blue"
      - hex: "#1f77b4"
      - RGB list/tuple: [r, g, b] where each value is in [0,1] or [0,255]
    Returns value normalized for matplotlib, or None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            rgb = [float(x) for x in value]
        except (TypeError, ValueError):
            raise ValueError(f"invalid color {value!r}")
        if any(v < 0 for v in rgb):
            raise ValueError(f"RGB color must be non-negative, got {value!r}")
        if any(v > 1.0 for v in rgb):
            if any(v > 255.0 for v in rgb):
                raise ValueError(f"RGB values >255 are not allowed: {value!r}")
            rgb = [v / 255.0 for v in rgb]
        return tuple(rgb)
    raise ValueError(
        "color must be a string/hex or RGB list like [0.1, 0.2, 0.3] or [25, 120, 200]"
    )


def format_frac(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def expand_runs_dir(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def find_max_metric(file_path: Path, metric_column: str) -> float | None:
    """Max over epochs; same rules as show_metrics.ipynb."""
    if not file_path.is_file():
        return None
    with open(file_path, mode="r", newline="") as csvfile:
        reader = csv.reader(x.replace("\0", "") for x in csvfile)
        headers = next(reader)
        headers = [x.lstrip() for x in headers]

        if metric_column == "fscore":
            precision_index = headers.index("metrics/precision(B)")
            recall_index = headers.index("metrics/recall(B)")
            max_metric_value = float("-inf")
            found = False
            for row in reader:
                try:
                    precision = float(row[precision_index].strip())
                    recall = float(row[recall_index].strip())
                    if precision + recall <= 0:
                        continue
                    fscore = 2 * precision * recall / (precision + recall)
                    max_metric_value = max(max_metric_value, fscore)
                    found = True
                except (ValueError, IndexError):
                    continue
            return max_metric_value if found else None

        try:
            column_index = headers.index(metric_column)
        except ValueError as e:
            raise ValueError(f"Column '{metric_column}' not found in CSV headers.") from e

        max_metric_value = float("-inf")
        found = False
        for row in reader:
            try:
                metric_value = float(row[column_index].strip())
                max_metric_value = max(max_metric_value, metric_value)
                found = True
            except (ValueError, IndexError):
                continue
        return max_metric_value if found else None


def default_fractions(kind: str) -> tuple[float, ...]:
    k = kind.strip().lower()
    if k == "random":
        return DEFAULT_RANDOM_FRACS
    if k == "al":
        return DEFAULT_AL_FRACS
    raise ValueError(f"chain kind must be 'random' or 'al', got {kind!r}")


def load_chain_defs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    chains = cfg.get("chains")
    if not chains:
        raise ValueError("config must contain a non-empty 'chains' list")
    out = []
    for i, c in enumerate(chains):
        if "template" not in c:
            raise ValueError(f"chains[{i}] missing 'template'")
        if "label" not in c:
            raise ValueError(f"chains[{i}] missing 'label'")
        if "kind" not in c:
            raise ValueError(f"chains[{i}] missing 'kind' (random or al)")
        out.append(c)
    return out


def series_for_chain(
    chain: dict[str, Any],
    runs_dir: Path,
    metric: str,
) -> tuple[list[float], list[float | None], list[str]]:
    """Returns (fractions, values, run_names) for plotting; None values skipped by plot."""
    kind = str(chain["kind"])
    template: str = chain["template"]
    fracs = chain.get("fractions")
    if fracs is not None:
        fractions = tuple(float(x) for x in fracs)
    else:
        fractions = default_fractions(kind)

    xs: list[float] = []
    ys: list[float | None] = []
    names: list[str] = []
    for frac in fractions:
        frac_s = format_frac(frac)
        run_name = template.replace("{frac}", frac_s)
        path = runs_dir / run_name / "results.csv"
        v = find_max_metric(path, metric)
        xs.append(frac)
        ys.append(v)
        names.append(run_name)
    return xs, ys, names


def make_chain_figure(
    series: list[tuple[str, list[float], list[float | None], Any]],
    metric: str,
    title: str | None = None,
):
    """
    Build the comparison figure. Safe to import from Jupyter (uses default matplotlib backend).
    Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    linestyles = ["-", "--", "-.", ":"]
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, (label, xs, ys, color) in enumerate(series):
        x_plot = [x for x, y in zip(xs, ys) if y is not None]
        y_plot = [y for x, y in zip(xs, ys) if y is not None]
        if not x_plot:
            print(f"[warn] no data for chain {label!r}", file=sys.stderr)
            continue
        c = normalize_color(color) if color is not None else colors[i % len(colors)]
        ax.plot(
            x_plot,
            y_plot,
            ls=linestyles[i % len(linestyles)],
            marker="o",
            c=c,
            label=label,
            alpha=0.9,
        )
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Train fraction")
    ylab = metric.replace("(B)", "").replace("metrics/", "")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_chains(
    series: list[tuple[str, list[float], list[float | None], Any]],
    metric: str,
    title: str | None,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig = make_chain_figure(series, metric, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def print_table(
    series: list[tuple[str, list[float], list[float | None], Any]],
    metric: str,
) -> None:
    all_fracs = sorted({x for _, xs, _, _ in series for x in xs})
    header = ["fraction"] + [s[0] for s in series]
    rows = [header]
    for frac in all_fracs:
        row = [format_frac(frac)]
        for label, xs, ys, _ in series:
            if frac not in xs:
                row.append("—")
                continue
            j = xs.index(frac)
            v = ys[j]
            row.append(f"{v:.4f}" if v is not None else "missing")
        rows.append(row)
    colw = [max(len(str(r[i])) for r in rows) for i in range(len(header))]
    for r in rows:
        print("  ".join(str(r[i]).ljust(colw[i]) for i in range(len(header))))
    print(f"(metric: {metric})")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Compare metric curves across training chains.")
    parser.add_argument("config", help="YAML config path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path (overrides config output)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Print table only, skip figure",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        print("Error: config must be a YAML mapping", file=sys.stderr)
        sys.exit(1)

    runs_dir = expand_runs_dir(cfg.get("runs_dir", "~/ultralytics/runs/detect"))
    metric = cfg.get("metric", "metrics/mAP50-95(B)")
    title = cfg.get("title")
    out = Path(os.path.expanduser(str(args.output or cfg.get("output", "chain_compare.png"))))

    chain_defs = load_chain_defs(cfg)
    series: list[tuple[str, list[float], list[float | None], Any]] = []
    for c in chain_defs:
        label = str(c["label"])
        color = c.get("color")
        xs, ys, names = series_for_chain(c, runs_dir, metric)
        series.append((label, xs, ys, color))
        missing = [(format_frac(x), n) for x, y, n in zip(xs, ys, names) if y is None]
        if missing:
            for fs, n in missing:
                print(f"[warn] {label}: missing {runs_dir / n / 'results.csv'} (frac {fs})", file=sys.stderr)

    print_table(series, metric)
    if not args.no_plot:
        plot_chains(series, metric, title, out)


if __name__ == "__main__":
    main()
