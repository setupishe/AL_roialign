from pathlib import Path
import shutil
import yaml
from tqdm import tqdm
import math


SIMILARITY_DATA_PATH = "/home/smarkov1001/NVI/scripts/TMP/similarity_pairs/similarity_data_sum_2nd_removal.yaml"
TRAIN_DATA_SOURCE = "/home/smarkov1001/sm_2x3080-1/outputs/human_head_detection_yolov8_dataset/train"
DESIRED_PERCENT_OF_SHITTY_IMAGES = 3 # %
ALLOWED_IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png")
DST_DIR_FOR_HIGH_FP_IMAGES = "/home/smarkov1001/sm_2x3080-1/outputs/human_head_detection_yolov8_dataset/remove_candidates_minus3/"

print("reading train files")
train_files = [
    file for file in Path(TRAIN_DATA_SOURCE).glob("*.*") 
    if (file.is_file() and (file.suffix in ALLOWED_IMAGE_SUFFIXES))
]
desired_amount_of_shitty_images = int(
    len(train_files)*(DESIRED_PERCENT_OF_SHITTY_IMAGES / 100)
)

print("loading similarity data yaml")
similarity_data = yaml.safe_load(Path(SIMILARITY_DATA_PATH).read_text())

image_to_max_similarity_index = {}
for train_e_path, similarity_index in tqdm(similarity_data.items()):
    original_image = None
    for suf in ALLOWED_IMAGE_SUFFIXES:
        original_image = str(
            Path(TRAIN_DATA_SOURCE) / f"{str(Path(train_e_path).name).split('_cropped')[0]}{suf}"
        )
        if Path(original_image).is_file():
            break
    if not Path(original_image).is_file():
        raise FileNotFoundError(f"origial image was not found for {train_e_path}")
    
    try:
        current_image_similarity_index = image_to_max_similarity_index[original_image]
    except KeyError:
        current_image_similarity_index =  -math.inf
    
    if current_image_similarity_index < similarity_index:
        image_to_max_similarity_index[original_image] = similarity_index

shitty_counter = 0
shitty_paths = []
for original_image in tqdm(
    sorted(image_to_max_similarity_index, key=image_to_max_similarity_index.get, reverse=True), 
    total=len(image_to_max_similarity_index)
):
    shitty_paths.append(Path(original_image).name)
    shitty_counter += 1
    if shitty_counter >= desired_amount_of_shitty_images:
        print(f"We found desired amount of images with high FP cumulative score: {desired_amount_of_shitty_images}")
        print("exiting loop")
        break

# with open("/home/smarkov1001/NVI/scripts/TMP/similarity_pairs/original_images_with_high_fp_score.yaml", 'w') as stream:
#     yaml.safe_dump(
#         {"images_from_train_with_high_fp_sum_score": sorted(shitty_paths)}, stream, indent=2, width=1000 
#     )

dst_img_dir = Path(DST_DIR_FOR_HIGH_FP_IMAGES)
dst_img_dir.mkdir(exist_ok=True, parents=True)
print("copying images with high fp cumulative score")
for img_path in tqdm(shitty_paths):
    src_path: Path = Path(TRAIN_DATA_SOURCE) / img_path
    dst_path: Path = dst_img_dir / img_path
    shutil.move(str(src_path), str(dst_path))

    corresponding_annotation = src_path.with_suffix(".txt")
    dst_annotation = dst_path.with_suffix(".txt")
    shutil.move(str(corresponding_annotation), str(dst_annotation))
    
    


