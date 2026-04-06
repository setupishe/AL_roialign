#!/bin/bash
# GPU 0: matryoshka_fast base → then MatrE AL chain (sequential)
python3 run_chain.py --skip-git-check configs/random_coco_s_scratch_matryoshka_fast_wm.yaml
python3 run_chain.py --skip-git-check configs/distance_matrE_fast_coco_s_scratch_wm.yaml
