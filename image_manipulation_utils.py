from typing import Union, Optional, List
from pathlib import Path
from utils import file_to_annotation, Annotation
from PIL import Image


@DeprecationWarning
def __black_out_image_except_bboxes(
        image_path: Union[str, Path], 
        labels_path: Optional[Union[str, Path]]=None,
        save_path: Optional[Union[str, Path]]=None
    ) -> None:
    if labels_path is None:
        labels_path = Path(image_path).with_suffix(".txt")
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels were not found at: {labels_path}")
    # TODO: use widening when cutting images, return widened bboxes
    image_annotations = file_to_annotation(labels_path)
    image = Image.open(image_path)
    blank_image = Image.new(mode='RGB', size=image.size, color='black')
    for annotation in image_annotations:
        bbox = annotation.bbox
        bbox = bbox.to_absolute(image.size)
        region = bbox.rectangle()
        cut_region = image.crop(region)
        blank_image.paste(cut_region, region)
    if save_path is not None:
        blank_image.save(save_path)


def black_out_image_except_bboxes(
        image_path: Union[str, Path],
        annotation_list: List[Annotation],
        save_dir: Union[str, Path]
    ) -> Path:
    image = Image.open(image_path)
    blank_image = Image.new(mode='RGB', size=image.size, color='black')
    for annotation in annotation_list:
        bbox = annotation.bbox
        #bbox = bbox.to_absolute(image.size)
        region = bbox.rectangle()
        cut_region = image.crop(region)
        blank_image.paste(cut_region, region)
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    dst_path = Path(save_dir) / Path(image_path).name
    blank_image.save(dst_path)
    return  dst_path



if __name__ == "__main__":
    pass