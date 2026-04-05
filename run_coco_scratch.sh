#!/bin/bash
set -e

python3 run_chain.py --skip-git-check configs/distance_coco_s_scratch.yaml
python3 run_chain.py --skip-git-check configs/distance_matrE_coco_s_scratch.yaml
