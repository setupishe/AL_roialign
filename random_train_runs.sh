#!/bin/bash

# Define the range values
ranges=(0.2 0.3 0.4 0.5 0.6 0.7)

# Loop through each range value
for range in "${ranges[@]}"
do
  echo "Running YOLO training with range: $range"
  yolo detect mode=train model=yolov8m.pt pretrained=False data=VOC_$range.yaml batch=48 name=VOC_random_$range epochs=65
done
