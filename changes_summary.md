# Changes Summary

Compared against commit `f6dab3bf676f1ab22040767657124694851f62fe`.

---

## `produce_detection_embeddings.py`

### Embedding strategies (default / separate / Matryoshka)

The producer now accepts a `strategy` parameter (`"default"`, `"separate"`, or any other string for Matryoshka mode):

- **default** – same as before: ROI-align each feature map, flatten, and concatenate into one vector.
- **separate** – returns a *list* of 3 independent per-map embeddings instead of one combined vector. Saved as `<name>.m0.npy`, `.m1.npy`, `.m2.npy`.
- **Matryoshka** – interleaves equal channel slices from all 3 maps so that any prefix of the final vector contains a proportional fraction of *every* map. Controlled by `matryoshka_slices` (number of blocks). Falls back to the largest common divisor if the preferred slice count doesn't evenly divide all channel counts. This mode is model-scale-agnostic (no longer hard-codes YOLOv8s channel ratios).

### Configurable ROI resolution

`EMBEDDING_TENSORS_HW_RESOLUTION_BEFORE_FLATTENING` moved from a class constant to an instance attribute. The constructor accepts an optional `embedding_tensors_hw_resolution_before_flattening` sequence; when omitted it defaults to `(12, 6)` for `"default"` strategy and `(3, 3)` otherwise.

### Stable layer output ordering

Added `_get_layer_output_tensor_names_in_netron_order()`. The ROI-alignment loop now iterates over layer outputs in the same fixed order as `netron_layer_names`, instead of relying on the dict iteration order of the ONNX session outputs.

### dtype fix for `roi_align`

Input tensors and ROI boxes are now explicitly cast to `float32` with matching dtypes, fixing a silent dtype mismatch that could occur with ONNX Runtime outputs.

---

## `preprocess_embedding_pool.py`

### `VectorDataset` accepts single `.npy` files

`_load_stored_embeddings` previously required directory paths. It now also handles a single `.npy` file path (or a list mixing files and directories). A docstring was added describing all supported formats.

### New `run_l2_normalization()` method

Applies per-vector L2 normalization and saves results — no dimensionality reduction. Used when `--skip-pca` is passed in the pipeline.

### New `run_standardization()` method

Applies per-batch zero-mean / unit-variance standardization and saves results — also skips PCA.

---

## `al_utils.py`

### HNSW backend

Added `build_hnsw_index()` using `hnswlib` (optional dependency). The index is batch-built and supports cosine space. `hnswlib` is imported with a try/except so Annoy-only setups still work.

### Matryoshka prefix slicing (`_slice_vector`)

New helper that truncates a vector to `use_dim` dimensions and L2-normalizes it. Both `build_annoy_index` and `build_hnsw_index` accept `use_dim` to build indices on a prefix of the full embedding — this is what makes the coarse-to-fine stages possible.

### Coarse-to-fine selection pipeline

`select_embeddings` now supports a `coarse_to_fine=True` mode with three stages:

1. **Stage 1** – build HNSW on `dim // d1_div` dimensions, rank all candidates, keep top `k × k1_mult`.
2. **Stage 2** – build HNSW on `dim // d2_div` dimensions, re-rank Stage-1 survivors, keep top `k × k2_mult`.
3. **Stage 3** – exact brute-force cosine KNN on full-dimensional vectors (via `sklearn.NearestNeighbors`), final ranking.

All multiplier/divisor values are configurable from the CLI.

### Density scoring mode

When `mode="density"`, the score is now computed as the inverse-square-distance sum over 5-NN (previously used 1-NN distance for both modes). Extracted into `_score_from_distances()`.

### Ordered deduplication

Selection results are now deduplicated into an *ordered* list (preserving the ranking) instead of a `set`, so downstream consumers see a deterministic priority order.

### `select_embeddings_voting()`

New function for the separate-maps workflow. Runs `select_embeddings` independently on each of 3 embedding spaces, then merges the per-space top-k lists by **voting**: images appearing in more spaces rank higher; ties are broken by average rank across spaces.

---

## `prepare_al_split.py`

### New CLI arguments

| Flag | Purpose |
|---|---|
| `--skip-pca` | Replace PCA with L2 normalization only |
| `--from-predictions` | Use YOLO predictions as bbox source instead of annotation `.txt` files |
| `--netron-layer-names` | Override the default 3 ONNX layer names |
| `--separate-maps-voting` | Produce 3 separate per-map embeddings and select via voting |
| `--roi-hw H W` | Override ROI Align output resolution |
| `--index-backend {annoy,hnsw}` | Choose ANN backend |
| `--coarse-to-fine` | Enable the 3-stage coarse-to-fine selection |
| `--ctf-k1-mult`, `--ctf-k2-mult` | Candidate multipliers for coarse-to-fine stages 1 & 2 |
| `--ctf-d1-div`, `--ctf-d2-div` | Dimension divisors for coarse-to-fine stages 1 & 2 |

### `_DONE.json` completion markers

Instead of comparing `.npy` file counts to decide whether to recompute embeddings or PCA, the pipeline now writes a `_DONE.json` file (with metadata) after each stage. Subsequent runs check for its presence, making skip-logic more robust (especially when using `--from-predictions` where the expected count isn't known upfront).

### `--from-predictions` support

When enabled, the script skips populating per-image annotation `.txt` files and lets `YoloEmbeddingsProducer` generate bboxes from model predictions. Cleanup also respects this flag.

### Separate-maps-voting integration

The entire pipeline (embedding production → preprocessing → file-list creation → selection) is branched to handle the per-map `.m{0,1,2}.npy` naming convention and to call `select_embeddings_voting` instead of `select_embeddings`.

### `--skip-pca` path

When set, runs `EmbeddingPoolPreprocessor.run_l2_normalization()` instead of the two-pass PCA workflow. For separate-maps mode, each map index is normalized independently (different embedding dimensions per map).

### Hardcoded paths updated

All `/ssd/temp/` paths changed to `/home/setupishe/datasets/`. YAML output path changed to an absolute path under `ultralytics/cfg/datasets/`.

### PCA path cleanup

Removed old commented-out subset-copying code. Added an `os.path.exists` guard before copying sidecar files (`.jpg`, `.txt`) during PCA subset creation.

---

## Quick Summary

- **Three embedding strategies** added to the producer: default (concat), separate (per-map), and Matryoshka (interleaved prefix-sliceable).
- **HNSW backend** added alongside Annoy for approximate nearest-neighbor search.
- **Coarse-to-fine 3-stage selection**: fast HNSW on reduced dims → HNSW on medium dims → exact cosine on full dims.
- **Voting-based multi-space selection**: when using separate per-map embeddings, each map votes independently and results are merged by vote count + average rank.
- **`--skip-pca`** option to replace PCA with simple L2 normalization.
- **`--from-predictions`** allows running the pipeline without pre-existing annotation files.
- **`_DONE.json` markers** replace fragile file-count checks for stage-skip logic.
- **ROI resolution and layer names** are now configurable from the CLI instead of being hardcoded.
- **dtype fix** for `roi_align` inputs and **stable layer ordering** in the embedding producer.
- **Ordered deduplication** in selection results (was unordered `set`, now ranked list).




