#!/bin/bash

# Define the range values
ranges=(0.2 0.3 0.4 0.5 0.6)

# Loop through each range value
for range in "${ranges[@]}"; do
  # Calculate the next range value using bc for floating-point arithmetic
  next_range=$(echo "$range + 0.1" | bc)
  
  folder_name="distance"
  if [[ "$range" == "0.2" ]]; then
    folder_name="random"
  fi

  # Output preparation message
  echo "PREPARING ON FRACTION $range FOR FRACTION 0$next_range"
  
  # Run the preparation script with the calculated arguments
  python3 prepare_al_split.py \
    --weights /home/setupishe/ultralytics/runs/detect/VOC_${folder_name}_$range/weights/best.pt \
    --from-fraction $range \
    --to-fraction 0$next_range \
    --from-split train_$range.txt \
    --dataset-name VOC \
    --split-name distance \
    --mode distance \
    --bg2all-ratio 0 \
    --cleanup
  
  # Output training message
  echo "TRAINING ON FRACTION 0$next_range"
  
  yolo detect mode=train \
  model=yolov8m.pt \
  pretrained=False \
  data=VOC_0${next_range}_distance.yaml \
  batch=48 \
  name=VOC_distance_0$next_range \
  epochs=65
done
