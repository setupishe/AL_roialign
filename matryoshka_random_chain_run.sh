# echo "Starting training on 20% of data..."
# yolo detect train data=VOC_0.2.yaml model=yolov8s.pt imgsz=640 batch=48 epochs=65 pretrained=False name=VOC_random_0.2_s_matryoshka matryoshka=True

# echo "Starting training on 30% of data..."
# yolo detect train data=VOC_0.3.yaml model=yolov8s.pt imgsz=640 batch=48 epochs=65 pretrained=False name=VOC_random_0.3_s_matryoshka matryoshka=True

# echo "Starting training on 40% of data..."
# yolo detect train data=VOC_0.4.yaml model=yolov8s.pt imgsz=640 batch=48 epochs=65 pretrained=False name=VOC_random_0.4_s_matryoshka matryoshka=True

# echo "Starting training on 50% of data..."
# yolo detect train data=VOC_0.5.yaml model=yolov8s.pt imgsz=640 batch=48 epochs=65 pretrained=False name=VOC_random_0.5_s_matryoshka matryoshka=True

# echo "Starting training on 60% of data..."
# yolo detect train data=VOC_0.6.yaml model=yolov8s.pt imgsz=640 batch=48 epochs=65 pretrained=False name=VOC_random_0.6_s_matryoshka matryoshka=True

echo "Starting training on 70% of data..."
yolo detect train data=VOC_0.7.yaml model=yolov8s.pt imgsz=640 batch=48 epochs=65 pretrained=False name=VOC_random_0.7_s_matryoshka matryoshka=True