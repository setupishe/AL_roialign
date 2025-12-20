#!/bin/bash

# Define the range values
ranges=(0.2 0.3 0.4 0.5 0.6)
device=0

# Coarse-to-fine candidate multipliers (override via env vars CTF_K1_MULT / CTF_K2_MULT)
ctf_k1_mult=${CTF_K1_MULT:-4}
ctf_k2_mult=${CTF_K2_MULT:-2}

# Optional: enable separate-maps selection with voting (override via env var SEPARATE_MAPS_VOTING=1)
separate_maps_voting=${SEPARATE_MAPS_VOTING:-0}

# Loop through each range value
for range in "${ranges[@]}"; do
  # Calculate the next range value using bc for floating-point arithmetic
  next_range=$(echo "$range + 0.1" | bc)
  
  folder_name="distance"
  if [[ "$range" == "0.2" ]]; then
    folder_name="random"
  fi

  fromsplit_name="_distance"
  if [[ "$range" == "0.2" ]]; then
    fromsplit_name=""
  fi
  # Output preparation message
  echo "PREPARING ON FRACTION $range FOR FRACTION 0$next_range"
  
  # Run the preparation script with the calculated arguments
  python3 prepare_al_split.py \
    --weights /home/setupishe/ultralytics/runs/detect/VOC_${folder_name}_${range}_s_matryoshka_everything_4_4_multipliers_1.25_3/weights/best.pt \
    --from-fraction $range \
    --to-fraction 0$next_range \
    --from-split train_${range}${fromsplit_name}.txt \
    --dataset-name VOC \
    --split-name distance_matryoshka_everything_4_4_multipliers_1.25_3 \
    --mode distance \
    --bg2all-ratio 0 \
    --device $device \
    --cleanup \
    --seg2line \
    --skip-pca \
    --index-backend hnsw \
    --netron-layer-names "/model.15/cv2/act/Mul /model.18/cv2/act/Mul /model.21/cv2/act/Mul" \
    --coarse-to-fine \
    --ctf-k1-mult $ctf_k1_mult \
    --ctf-k2-mult $ctf_k2_mult \
    --roi-hw 4 4 \
    $( [ "$separate_maps_voting" -eq 1 ] && echo "--separate-maps-voting" )
  
  # Output training message
  echo "TRAINING ON FRACTION 0$next_range"
  
  yolo detect mode=train \
  model=yolov8s.pt \
  pretrained=False \
  data=VOC_0${next_range}_distance_matryoshka_everything_4_4_multipliers_1.25_3.yaml \
  batch=48 \
  name=VOC_distance_0${next_range}_s_matryoshka_everything_4_4_multipliers_1.25_3 \
  device=$device \
  epochs=65 \
  matryoshka=True
done