# Matryoshka (“Matreshka”) YOLOv8 (Ultralytics fork) — What Changed vs “Vanilla” YOLOv8

This repository is a fork of **Ultralytics/Ultralytics YOLOv8** with an added **Matryoshka (“nested width”) training + evaluation mode** for detection.

The goal of this document is to provide a **precise, code-grounded diff narrative** that another LLM (or a reviewer) can use to determine whether the *specific thing implemented here* existed previously, and to help you write a paper with clear claims and reproducible details.

---

## Baseline and fork identity (for reproducibility)

- **Upstream baseline**: Ultralytics tag **`v8.2.90`** (present in this repo’s git history).
- **This fork/branch**: `feature/matryoshka-weights`
- **Current HEAD (at time of writing)**: `ac402199` (“Refine Matryoshka validation and inference handling”)
- **Remotes (as configured in this repo)**:
  - `origin` (fetch): Ultralytics upstream (`ultralytics/ultralytics`)
  - `fork`/`origin` (push): `setupishe/ultralytics_matryoshka`

If you need a deterministic code snapshot for a paper, cite **the commit hash** (e.g. `ac402199`) rather than “latest”.

### Matryoshka change timeline (commits on top of `v8.2.90`)

These are the Matryoshka-related commits as they appear in this repo:

- `0886dc05` — “added matryoshka variant”
- `73ec33e7` — “added weights”
- `324a7fc4` — “added shared preds assign flag”
- `211f4420` — “added weights schedule and removed normalizing”
- `ae99906e` — “Add Matryoshka BatchNorm freezing option”
- `28499394` — “Add Matryoshka weight warmup configuration”
- `65b9fcd6` — “Add Matryoshka validation support and backend reuse”
- `ac402199` — “Refine Matryoshka validation and inference handling”

---

## What “Matryoshka” means in *this* repo (high-level)

### Core idea

During training, the **detection head** is executed multiple times at different **channel granularities** (nested subsets of feature channels), producing multiple prediction sets from the *same* model weights.

The granularities are (per detection layer, i.e., per feature map):

- **1/8 width** (use first \(C/8\) channels)
- **1/4 width** (use first \(C/4\) channels)
- **1/2 width** (use first \(C/2\) channels)
- **full width** (use all \(C\) channels)

All these granularities share the same head weights; the only difference is that the head **slices** the input feature channels and correspondingly slices the **input-channel dimension** of the first conv’s kernel weights.

### What is and isn’t “slimmed”

- **Backbone/neck**: unchanged compute and unchanged tensor widths.
- **Detection head**: optionally runs using only a prefix of the input feature channels for its first conv in each branch (box/regression branch `cv2`, class branch `cv3`).

This is best described as a **Matryoshka / nested-width detection head**, not a full-network slimmable YOLOv8 (unless you additionally modify the backbone/neck).

---

## Key behavioral differences vs vanilla Ultralytics YOLOv8

### 1) Detect head supports Matryoshka multi-granularity outputs (training) and selectable granularity (inference)

#### Training behavior change

If `matryoshka=True` **and** the model is in training mode, the Detect head returns:

- **Type**: `List[List[Tensor]]`
- **Meaning**: `preds[granularity_idx][det_layer_idx]`
- **Granularity order**: `[1/8, 1/4, 1/2, 1]` (full width is last)

Each `Tensor` is the standard YOLOv8 per-layer head output: shape `[B, no, H, W]` where `no = nc + 4*reg_max`.

This is a breaking change vs vanilla training outputs (which are typically `List[Tensor]` for detection heads).

#### Inference behavior change

At inference time, the Detect head supports an attribute:

- `matryoshka_infer_idx`:
  - `-1` (default): standard full-width behavior (vanilla)
  - `0/1/2/3`: select granularity \(1/8, 1/4, 1/2, 1\) respectively

If `matryoshka_infer_idx != -1`, inference runs the head at the chosen granularity by slicing:

- the input feature map channels (`x[i][:, :g, :, :]`)
- the first conv’s weights (`conv.weight[:, :g, :, :]`)

Importantly, this selector is implemented so it can also run on “vanilla” checkpoints by computing granularities on the fly from `x[i].shape[1]` (even if the checkpoint wasn’t trained with Matryoshka).

#### BN “auxiliary freeze” option

There is an optional mechanism to **freeze BatchNorm running-stat updates** for *auxiliary* granularities (all but full width) during training:

- `matryoshka_bn_aux_freeze=True` causes BN layers in the head branches to temporarily use **momentum=0.0** for auxiliary passes.
- Full-width pass keeps normal BN behavior.

Rationale: auxiliary passes would otherwise update running stats multiple times per batch using different channel slices, which can destabilize BN statistics and/or make them inconsistent across granularities.

**Primary implementation file**: `ultralytics/nn/modules/head.py` (`Detect`).

---

### 2) New Matryoshka-aware detection loss wrapper

Vanilla YOLOv8 uses `v8DetectionLoss` directly for detection training.

This fork adds a wrapper that can:

- consume either vanilla predictions or Matryoshka list-of-lists predictions,
- compute a loss per granularity,
- combine them with configurable weights,
- optionally reuse the same assignment across granularities.

#### New module

- `ultralytics/nn/modules/loss_mrl.py`
  - class: `MatryoshkaDetectionLoss`
  - internally uses: `ultralytics.utils.loss.v8DetectionLoss`

#### Weighting behavior

New config key:

- `matryoshka_weights`:
  - `None` → defaults to equal weights (vector of ones)
  - if provided, it should be length = number of granularities produced (typically 4)

Notably, the weights are **not normalized** in the current implementation (“preserves full-width gradient scale”).

#### Optional auxiliary-weight warmup schedule

New config keys:

- `matryoshka_weight_warmup` (bool)
- `matryoshka_weight_warmup_steps` (int)
- `matryoshka_weight_warmup_start_step` (int)

Behavior: the warmup scales **auxiliary** weights (all but the last/full-width granularity) from 0 → 1 over a number of loss calls, optionally after a start delay. The full-width weight is not altered.

#### Optional shared assignment across granularities

New config key:

- `matryoshka_shared_assign` (bool)

If enabled (and multiple granularities exist), the assigner is computed **once** using the **full-width predictions**, then the same `target_scores/fg_mask/target_bboxes` are reused to compute bbox/cls/dfl losses for each granularity.

Rationale:

- reduces compute (assignment can be expensive),
- enforces consistent matching across granularities (important when you want nested heads to be comparable).

---

### 3) Model parsing + stride building updated to handle Matryoshka outputs

#### Model construction passes Matryoshka flag to `Detect`

In the model YAML parsing (`parse_model`), `Detect` receives an extra argument: `matryoshka` (bool).

This is what actually turns on Matryoshka behavior inside `Detect`.

#### Stride computation handles list-of-lists output

During model initialization, YOLOv8 runs a dummy forward pass to infer strides.

Matryoshka training outputs are nested; for stride computation this fork takes one granularity’s output (spatial dims are identical across granularities) and proceeds normally.

**Primary implementation file**: `ultralytics/nn/tasks.py` (`DetectionModel`, `parse_model`).

---

### 4) Training plumbing: enabling Matryoshka and setting BN-freeze option

The detection trainer constructs the model with `matryoshka=self.args.matryoshka` and, when enabled, propagates BN-freeze preference to the Detect head:

- `matryoshka=True` enables Matryoshka behavior.
- `matryoshka_bn_aux_freeze=True` activates BN running-stat freeze for auxiliary passes.

**Primary implementation file**: `ultralytics/models/yolo/detect/train.py`.

---

### 5) Validation supports multi-granularity evaluation in one command

This fork adds a “run validation repeatedly at multiple widths” feature for detection validation (non-training mode).

New validation args:

- `matryoshka_val` (bool): if `True`, run validation at each requested fraction and return a flattened dict of results
- `matryoshka_val_fracs` (string): comma-separated fractions; supported mapping:
  - `0.125 → idx 0`
  - `0.25 → idx 1`
  - `0.5 → idx 2`
  - `1.0 → idx 3`

Behavior:

- The validator builds (or reuses) a single `AutoBackend` once.
- It locates the Detect head and sets `head.matryoshka_infer_idx` per fraction.
- It runs the normal validation logic repeatedly without reloading weights.
- It returns results under keys like `matryoshka/0.5/metrics/mAP50-95(B)` (etc).

**Primary implementation file**: `ultralytics/models/yolo/detect/val.py`.

---

### 6) Base validator allows reusing a prebuilt backend (enables fast repeated Matryoshka eval)

In vanilla Ultralytics, calling `val()` repeatedly typically reconstructs the backend.

This fork modifies `BaseValidator.__call__` to accept a pre-constructed `AutoBackend` instance and reuse it.

This is a small but important infrastructure change that makes `matryoshka_val=True` practical without repeated model loads.

**Primary implementation file**: `ultralytics/engine/validator.py`.

---

### 7) YOLO `.train()` forwards `matryoshka=` from kwargs into trainer args

This fork explicitly forwards `matryoshka` from `YOLO(...).train(matryoshka=True, ...)` into the constructed training args dict.

**Primary implementation file**: `ultralytics/engine/model.py`.

---

## New/changed configuration keys (defaults in this repo)

These defaults are added to the global default config dictionary (`DEFAULT_CFG_DICT`):

- **`matryoshka`**: `False`
  - Enables Matryoshka behavior in the Detect head + Matryoshka loss wrapper.
- **`matryoshka_weights`**: `None`
  - Optional per-granularity weights (length should match number of granularities produced; typically 4).
- **`matryoshka_shared_assign`**: `False`
  - If `True`, compute the assignment once from full-width predictions and reuse it for all granularities.
- **`matryoshka_bn_aux_freeze`**: `False`
  - If `True`, freeze BN running-stat updates for auxiliary granularities in the Detect head.
- **`matryoshka_weight_warmup`**: `False`
  - Enable auxiliary-weight warmup (ramps only auxiliary weights; full-width unchanged).
- **`matryoshka_weight_warmup_steps`**: `0`
  - Warmup duration in loss calls.
- **`matryoshka_weight_warmup_start_step`**: `0`
  - Delay before warmup starts (loss calls).
- **`matryoshka_val`**: `False`
  - If `True`, run multi-granularity validation.
- **`matryoshka_val_fracs`**: `"0.125,0.25,0.5,1.0"`
  - Fractions to evaluate for `matryoshka_val=True`.

**Primary implementation file for defaults**: `ultralytics/utils/__init__.py`.

---

## How to use (practical recipes)

### Train a Matryoshka detection model

Example CLI (detection):

- `yolo train task=detect model=yolov8n.yaml data=coco.yaml matryoshka=True`

Common Matryoshka-specific knobs:

- `matryoshka_shared_assign=True` (reuses assignment from full width)
- `matryoshka_bn_aux_freeze=True` (stabilizes BN running stats)
- `matryoshka_weights=[w0,w1,w2,w3]` (weights for `[1/8,1/4,1/2,1]`)
- `matryoshka_weight_warmup=True matryoshka_weight_warmup_steps=...`

### Evaluate multiple granularities in one run

- `yolo val task=detect model=path/to/weights.pt data=... matryoshka_val=True`
- Optional: `matryoshka_val_fracs=0.25,0.5,1.0`

Output keys are prefixed with `matryoshka/<frac>/...`.

### Inference at a chosen width fraction (programmatic)

Conceptually:

- Load model as usual.
- Find the Detect head (typically `model.model[-1]` for PyTorch models).
- Set `head.matryoshka_infer_idx` to `0/1/2/3` and run prediction.

This selector works even for non-Matryoshka-trained weights, but you should expect degraded accuracy unless trained for it.

---

## File-by-file change map (vs Ultralytics v8.2.90)

The git diff vs `v8.2.90` touches 15 files. Most of the “large” diffs are formatting-only changes; the Matryoshka feature is concentrated in the files below.

### Matryoshka feature implementation (core)

- **`ultralytics/nn/modules/head.py`**
  - `Detect(..., matryoshka=False, matryoshka_bn_aux_freeze=False)`
  - multi-granularity training forward (returns list-of-lists)
  - inference selector `matryoshka_infer_idx`
  - optional BN running-stat freeze for auxiliary passes

- **`ultralytics/nn/modules/loss_mrl.py`** (new)
  - `MatryoshkaDetectionLoss` wrapper around `v8DetectionLoss`
  - granularity weighting, optional weight warmup
  - optional shared assignment from full-width predictions

- **`ultralytics/nn/tasks.py`**
  - `DetectionModel(..., matryoshka=False)`
  - stride inference handles nested outputs
  - `init_criterion()` returns `MatryoshkaDetectionLoss` when enabled
  - `parse_model(..., matryoshka=...)` forwards matryoshka flag into `Detect`

- **`ultralytics/models/yolo/detect/train.py`**
  - trainer constructs `DetectionModel(..., matryoshka=self.args.matryoshka)`
  - passes BN-freeze preference to Detect head

- **`ultralytics/models/yolo/detect/val.py`**
  - `matryoshka_val` multi-pass evaluation across fractions via `matryoshka_infer_idx`

- **`ultralytics/engine/validator.py`**
  - permits backend reuse (`AutoBackend`) to avoid reloads across repeated eval passes

- **`ultralytics/engine/model.py`**
  - forwards `matryoshka` kwarg into training args

- **`ultralytics/utils/__init__.py`**
  - adds Matryoshka defaults into `DEFAULT_CFG_DICT`

### Other (non-Matryoshka) repo changes you should be aware of

These changes are present in the repo diff vs upstream tag, but are not essential to the Matryoshka head/loss concept:

- **`ultralytics/cfg/datasets/VOC.yaml`**
  - changed VOC train/val specification to use `train.txt` / `val.txt`.

- **`ultralytics/cfg/datasets/coco_0.3_confidences.txt`**
  - custom COCO dataset variant pointing `train:` to `train2017_0.3_confidences.txt`.

- **`ultralytics/data/loaders.py`**, **`ultralytics/cfg/__init__.py`**, **`ultralytics/utils/loss.py`**, **`ultralytics/utils/metrics.py`**, **`.gitignore`**
  - appear largely formatting/refactor oriented in this diff; review if your paper claims “only Matryoshka changes” and you need a strict minimal patch.

---

## What to tell another chatbot/reviewer to look for (novelty search guidance)

If you want another system to decide “was this done before?”, give it the **specific mechanism** and **keywords** that match the implementation:

### The mechanism (most specific phrasing)

- “YOLOv8 detection head trained with **nested channel prefixes** (1/8, 1/4, 1/2, 1) using the **same head weights**, where each auxiliary head pass slices the input channels and slices the first conv kernel’s input-channel dimension.”
- “Loss computed across all granularities with optional **shared assignment** computed from the full-width predictions and reused for all smaller widths.”
- “Optional **BN running-stat freeze** for auxiliary granularities by setting BN momentum to 0.0 only for auxiliary passes.”
- “Validation runs multiple passes by setting a **head-level inference selector** `matryoshka_infer_idx`.”

### Useful keywords for literature / code search

- **Slimmable networks**, **nested subnetworks**, **dynamic width**
- **Once-for-All (OFA)** / **progressive shrinking** / **sandwich rule**
- **Matryoshka representation learning** (name overlap; verify whether concept matches)
- **multi-exit / early-exit** (related but different: exits vs width slicing)
- “shared assigner” / “task-aligned assigner reuse” across subnetworks

### Where the “paper novelty” likely lives in this implementation

- The design choice to apply Matryoshka only to the **YOLO head input-channel dimension** (minimal architectural disruption).
- The **shared assignment** option (forces consistent matching and reduces compute).
- The BN auxiliary freeze scoped only to auxiliary passes (practical training stabilization).
- The evaluation harness (`matryoshka_val`) that reports per-width metrics cleanly.

---

## Repro/ablation checklist (paper-friendly)

To support a paper claim, you likely want the following experiments:

- **Baseline**: upstream YOLOv8 (same tag, same data, same hyperparams).
- **Matryoshka full-width**: `matryoshka=True` but evaluate at `1.0` (idx 3) to see if training hurts/helps main model.
- **Matryoshka sub-width performance**: evaluate at `0.5`, `0.25`, `0.125`.
- **Ablate `matryoshka_shared_assign`**: on vs off.
- **Ablate `matryoshka_bn_aux_freeze`**: on vs off.
- **Ablate weight schedules**: fixed weights vs warmup.

---

## Minimal “Matryoshka-only patch” recommendation (if you need it)

If you want to publish a clean patch (for reviewers), a good minimal set is:

- `ultralytics/nn/modules/head.py`
- `ultralytics/nn/modules/loss_mrl.py`
- `ultralytics/nn/tasks.py`
- `ultralytics/models/yolo/detect/train.py`
- `ultralytics/models/yolo/detect/val.py`
- `ultralytics/engine/validator.py`
- `ultralytics/engine/model.py`
- `ultralytics/utils/__init__.py`

Everything else can be dropped or cherry-picked based on whether you need it for your experimental pipeline.


