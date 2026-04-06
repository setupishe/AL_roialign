#!/bin/bash
python3 run_chain.py --skip-git-check configs/distance_voc_scratch_s_v3_cont.yaml
python3 run_chain.py --skip-git-check configs/random_voc_s_scratch_matryoshka_fast.yaml
