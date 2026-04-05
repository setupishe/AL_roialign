#!/bin/bash
set -e

python3 run_chain.py --skip-git-check configs/distance_voc_scratch_s_pseudo.yaml
python3 run_chain.py --skip-git-check configs/distance_matrE_voc_s_scratch_pseudo.yaml
