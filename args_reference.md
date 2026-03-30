# Args Reference

## `prepare_al_split.py` (CLI)

Defined in `prepare_al_split.py` (argparse). Several arguments have no `required=True` flag but are still needed at runtime.

### Positional-style required

| Arg | Notes |
|---|---|
| `--from-fraction` | |
| `--to-fraction` | |
| `--from-split` | |
| `--weights` | `.pt` path; script reads `best_conf.txt` two levels up |
| `--device` | CUDA device id for ONNX provider |
| `--split-name` | |
| `--dataset-name` | |
| `--bg2all-ratio` | |

### Optional (defaults)

| Arg | Default | Notes |
|---|---|---|
| `--mode` | `distance` | |
| `--index-backend` | `annoy` | `annoy` \| `hnsw` |
| `--granularity-divs` | `8 4 2 1` | space-separated; used for `matryoshka_variance` selection |
| `--ctf-k1-mult` | `4.0` | coarse-to-fine |
| `--ctf-k2-mult` | `2.0` | coarse-to-fine |
| `--ctf-d1-div` | `8` | coarse-to-fine |
| `--ctf-d2-div` | `4` | coarse-to-fine |
| `--onnx-batch-size` | `16` | |
| `--io-workers` | `4` | |
| `--seed` | `42` | random seed for background sampling |
| `--roi-hw` / `--embedding-hw` | *(none)* | two ints: **H W** (metavar order) |
| `--datasets-dir` | `/home/setupishe/datasets` | |
| `--ultralytics-cfg-dir` | `/home/setupishe/ultralytics/ultralytics/cfg/datasets` | |
| `--netron-layer-names` | *(built-in 3 concat layers)* | space-separated ONNX node names |

### Flags

| Flag | Notes |
|---|---|
| `--cleanup` | remove caches / lists after success |
| `--seg2line` | segmentation labels → bbox lines |
| `--skip-pca` | L2-normalize only (no PCA) |
| `--from-predictions` | bboxes from model, not label `.txt` in train image dir |
| `--separate-maps-voting` | three maps, vote to combine |
| `--coarse-to-fine` | |
| `--no-batched-inference` | ONNX batch dim fixed to 1 |
| `--save-crops` | write crop JPGs beside `.npy` |

---

## YAML `prepare_args` → `run_chain.py`

Only these keys are accepted in `prepare_args`; anything else makes `run_chain.py` raise `ValueError`. They map to `prepare_al_split.py` flags (`_` → `-`), except `batched_inference: false` → `--no-batched-inference`.

**Boolean (truthy adds flag, except `batched_inference`):**

`batched_inference`, `save_crops`, `seg2line`, `cleanup`, `skip_pca`, `coarse_to_fine`, `from_predictions`, `separate_maps_voting`

**Value (becomes `--key value`):**

`index_backend`, `io_workers`, `onnx_batch_size`, `granularity_divs`, `ctf_k1_mult`, `ctf_k2_mult`, `ctf_d1_div`, `ctf_d2_div`, `netron_layer_names`, `seed`

**Special:**

`roi_hw: [H, W]` → `--roi-hw H W`

`conf_criteria.py` chains only forward `seg2line` and `cleanup` from this set.

CLI-only options (`--datasets-dir`, `--ultralytics-cfg-dir`, etc.) are **not** wired through YAML in `run_chain.py`.

---

## `yolo train` args from `configs/*.yaml`

`run_chain.py` builds `yolo train key=value ...` from the `yolo_args` dict (template expansion for `${...}`).

### Keys that appear in at least one config

`batch`, `data`, `deterministic`, `device`, `epochs`, `imgsz`, `matryoshka`, `matryoshka_bn_aux_freeze`, `matryoshka_shared_assign`, `matryoshka_weight_warmup`, `matryoshka_weight_warmup_start_step`, `matryoshka_weight_warmup_steps`, `matryoshka_weights`, `model`, `name`, `pretrained`, `seed`

### Usually the same across experiments

- `batch`: `48`
- `epochs`: `65`
- `pretrained`: string `"False"`

### Often varied

| Key | Examples in repo |
|---|---|
| `model` | `yolov8s.pt`, `yolov8m.pt` |
| `data` | `VOC_${next_range}_${split_name}.yaml`, `VOC_${fraction}.yaml`, `COCO_${fraction}.yaml`, `${DATASET_NAME}_...` |
| `name` | run folder under `ultralytics/runs/detect/` (templates include fraction / mode / matryoshka tags) |
| `device` | `chain` configs often use `${device}` from top-level `device:`; **omitted** in `configs/confidence.yaml` and all `type: random_train` configs (Ultralytics default) |
| `imgsz` | `640` in `random_train` configs only |
| `seed` | `0`, `1`, `2` where specified |
| `deterministic` | `"False"` where specified |
| Matryoshka-related | `matryoshka`, `matryoshka_shared_assign`, `matryoshka_bn_aux_freeze`, warmup steps; `matryoshka_weights` as `"1.0, 0.8, 0.4, 0.2"` or `"[0.2,0.3,0.4,1.0]"` |

String casing for boolean-like matryoshka flags is inconsistent in YAML (`"true"` vs `"True"`); both are passed through to the CLI as written.
