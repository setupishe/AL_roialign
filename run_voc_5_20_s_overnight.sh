#!/usr/bin/env bash
set -uo pipefail

cd /home/setupishe/bel_conf || exit 1
export VIRTUAL_ENV="/home/setupishe/bel_conf/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

run_cfg() {
  local cfg="$1"
  echo
  echo "=== RUNNING ${cfg} ==="
  yes y | "${VIRTUAL_ENV}/bin/python" run_chain.py "${cfg}"
}

# run_cfg configs/random_voc_baseline_5_20_s_night.yaml
# run_cfg configs/random_voc_matryoshka_everything_really_everything_5_20_s_night.yaml
run_cfg configs/distance_5_20_s_night.yaml
run_cfg configs/distance_matryoshka_everything_really_everything_5_20_s_night.yaml
