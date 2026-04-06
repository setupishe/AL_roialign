#!/bin/bash
# GPU 1: confidence chain (needs COCO_random_0.2_s_scratch to exist first)
python3 run_chain.py --skip-git-check configs/confidence_coco_s_scratch_wm.yaml
