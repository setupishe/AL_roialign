from pathlib import Path
from PIL import Image, ImageDraw
from utils import file_to_annotation
from tqdm import tqdm


ALLOWED_IMAGE_SUFFIXES = ".jpeg", ".jpg", ".png"

root = "/home/smarkov1001/sm_2x3080-1/outputs/human_head_detection_yolov8_dataset/remove_candidates"
dst = "/home/smarkov1001/NVI/scripts/TMP/similarity_pairs/remove_candidates_VIS"

files = [
    file for file in Path(root).glob("*.*")
    if (
        file.is_file()
        and
        (file.suffix in ALLOWED_IMAGE_SUFFIXES)
    )
]

Path(dst).mkdir(exist_ok=True, parents=True)
for file in tqdm(files):
    annotations = file_to_annotation(file.with_suffix(".txt"))
    image = Image.open(file)
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        draw.rectangle(
            annotation.bbox.to_absolute(image.size).rectangle(),
            outline='purple' if annotation.class_index == 0 else 'green',
            width = 2
        )
    image.save(Path(dst) / file.name)
    

