# 8xH100 Transfer & Execution Guide

This document outlines the necessary steps to prepare datasets and execute the PhD experiments on the 8xH100 bare-metal machine.

## 1. Dataset Preparation (Pre-Transfer)

Before transferring the datasets to the 8xH100 machine, you must download and preprocess the COCO dataset using the Ultralytics utility. This avoids downloading and processing on the bare-metal machine itself.

### Steps to prepare COCO locally:
1. Install the `ultralytics` package on your local machine:
   ```bash
   pip install ultralytics
   ```
2. Download and convert the COCO dataset. The `convert_coco` utility will download the annotations and images, and convert them to the YOLO format:
   ```python
   from ultralytics.data.converter import convert_coco
   
   # This will download the dataset to your current working directory 
   # or the default Ultralytics datasets_dir
   convert_coco(labels_dir='../datasets/coco/annotations', use_segments=False, use_keypoints=False, cls91to80=True)
   ```
   *(Alternatively, if you already have the COCO images and annotations downloaded, just point `labels_dir` to the annotations folder.)*
3. Once the dataset is in the YOLO format (with `images/` and `labels/` directories), package it and transfer it to the 8xH100 machine.

## 2. Environment Setup

Once the datasets are transferred, run the setup script to configure the Conda environment and install the required packages:

```bash
bash setup_8xh100.sh
```
You will be prompted to enter the absolute path to the directory containing the transferred datasets.

## 3. GPU Execution Context & Warnings

Based on empirical benchmarks and the specific architecture of the experiments, please adhere to the following guidelines when running on the 8xH100 machine:

### Bare-Metal Session Management
Since this is a bare-metal machine, SSH disconnects will kill running processes. To prevent the GPUs from "going down when idle" or losing a training run:
- **Always use `tmux` or `screen`** to manage your terminal sessions.
- Example `tmux` workflow:
  ```bash
  tmux new -s experiment1
  conda activate phd_env
  python run_chain.py ...
  # Detach with Ctrl+B, then D
  ```

### DDP (Distributed Data Parallel) Anomaly
- **Avoid `batch=64` when using DDP.** There is a known anomaly where 2x GPUs at `batch=64` (32 per GPU) runs slower than a single GPU due to how the nominal batch size (`nbs=64`) interacts with gradient accumulation in Ultralytics.
- **Recommendation:** Use `batch=48` for DDP runs.

### RAM Bottlenecks with Parallel Chains
- If you are running multiple independent experiment chains in parallel on different GPUs, the dataloader workers can consume a massive amount of system RAM (prefetching image batches).
- **Recommendation:** Set `workers=4` in your training config or command line arguments to halve the RAM usage per chain without bottlenecking the compute-bound GPUs.

---
*Note: All experiments are strictly locked to `batch=48` unless otherwise specified for a baseline comparison.*
