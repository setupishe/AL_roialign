python3 prepare_al_split.py --from-fraction 0.6 --to-fraction 0.7 --weights /home/setupishe/ultralytics/runs/detect/active_0.6/weights/best.pt --conf 0.305 --cleanup
yolo detect mode=train model=yolov8m.pt pretrained=False data=coco_0.7_active.yaml batch=48 name=active_0.7 epochs=65