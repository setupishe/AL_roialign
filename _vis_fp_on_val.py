from pathlib import Path
from tqdm import tqdm
from utils import file_to_annotation, Bbox
from PIL import Image, ImageDraw

yolo_run_on_val_labels_folder = "/home/smarkov1001/NVI/scripts/TMP/Lev_task_140224/yolo_run64_on_val/labels"
val_label_files = [file for file in Path(yolo_run_on_val_labels_folder).glob("*.txt")]

gt_val_dir = Path("/home/smarkov1001/sm_3.46/outputs/human_head_detection_yolov8_dataset/val")
ground_truth_files = [
    gt_val_dir / file.name for file in val_label_files
]

assert all(file.is_file() for file in ground_truth_files)

sufs = (".jpeg", ".jpg", ".png")
val_images = []
for file in ground_truth_files:
    for suf in sufs:
        corresponding_image = gt_val_dir / file.with_suffix(suf).name
        if corresponding_image.is_file():
            break
    if not corresponding_image.is_file():
        raise FileNotFoundError
    val_images.append(corresponding_image)

print(f"{len(val_images)=}")
print(f"{len(ground_truth_files)=}")

for idx, val_label_file in enumerate(tqdm(val_label_files)):
    ground_truth_file = ground_truth_files[idx]
    val_image= val_images[idx]
    gt_annotations = file_to_annotation(ground_truth_file)
    prediction_annotations = file_to_annotation(val_label_file)
    
    real_fps = []
    for p in prediction_annotations:
        if p.class_index != 1:
            continue

        this_is_fp = True
        for gt in gt_annotations:
            if gt.class_index != 1:
                continue

            if Bbox.IoU(p.bbox, gt.bbox) > 0.05:
                this_is_fp = False
                break
        
        if this_is_fp and (p.confidence >= 0.47):
            real_fps.append(p)
    
    if len(real_fps) != 0:
        image = Image.open(val_image)
        draw = ImageDraw.Draw(image)
        for idx, fp_annotation in enumerate(real_fps):
            draw.rectangle(fp_annotation.bbox.to_absolute(image.size).rectangle(), outline='red', width=2)
        dst_path = f"/home/smarkov1001/NVI/scripts/TMP/Lev_task_140224/val_vis/{idx}_{val_image.name}"
        Path(dst_path).parent.mkdir(exist_ok=True, parents=True)
        image.save(dst_path)
