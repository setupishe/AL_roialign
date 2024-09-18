from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, List, Tuple, Union, Iterable, Sequence, Dict
from pathlib import Path
from PIL import Image
from functools import partial


ALLOWED_IMAGE_SUFFIXES = ".jpeg", ".jpg", ".png"


@dataclass
class Bbox:

    xc: float
    yc: float
    w: float
    h: float

    @cached_property
    def x_min(self) -> float:
        return self.xc - self.w / 2
    
    @cached_property
    def y_min(self) -> float:
        return self.yc - self.h / 2
    
    @cached_property
    def x_max(self) -> float:
        return self.xc + self.w / 2
    
    @cached_property
    def y_max(self) -> float:
        return self.yc  + self.h / 2

    def area(self) -> float:
        return abs(self.w * self.h)
    
    def normalize(self, image_wh: Tuple[int, int]) -> Bbox:
        img_w, img_h = image_wh
        xc = self.xc / img_w
        yc = self.yc / img_h
        w = self.w / img_w
        h = self.h / img_h
        return Bbox(xc, yc, w, h)
    
    def to_absolute(self, image_wh: Tuple[int, int]) -> Bbox:
        img_w, img_h = image_wh
        xc = self.xc * img_w
        yc = self.yc * img_h
        w = self.w * img_w
        h = self.h * img_h
        return Bbox(xc, yc, w, h)

    def rectangle(self) -> List[int]:
        ret = [self.x_min, self.y_min, self.x_max, self.y_max]
        ret = list(map(int, ret))
        if all(item == 0 for item in ret):
            raise ValueError("All elements of rectangle are zero. Did you forgot to switch to absolute coordinates?")
        return ret

    def clip(self, image_wh: Tuple[int, int]) -> Bbox:
        w, h = image_wh
        x_min = int(max(0, self.x_min))
        x_max = int(min(w, self.x_max))
        y_min = int(max(0, self.y_min))
        y_max = int(min(h, self.y_max))
        if all(item == 0 for item in (x_min, x_max, y_min, y_max)):
            raise ValueError(
                "All border coordinates of Bbox are zero. Did you forgot to switch to absolute coordinates before clip?"
            )   
        xc = (x_min + x_max) / 2
        yc = (y_min + y_max) / 2
        bw = x_max - x_min
        bh = y_max - y_min
        return Bbox(xc, yc, bw, bh)


    def is_normalized(self) -> bool:
        f = partial(round, ndigits=7)
        return all(
            0 <= attr <= 1 for attr in (
                list(self.__dict__.values()) + list(map(f, [self.x_min, self.y_min, self.x_max, self.y_max]))
                )
        )
    
    @staticmethod
    def overlap_area(this: Bbox, other: Bbox) -> float:
        dx = min(this.x_max, other.x_max) - max(this.x_min, other.x_min)
        dy = min(this.y_max, other.y_max) - max(this.y_min, other.y_min)
        if (dx>=0) and (dy>=0):
            return dx*dy     
        return 0.0  
    
    @staticmethod
    def IoU(this: Bbox, other: Bbox) -> float:
        return Bbox.overlap_area(this, other)/ (this.area() + other.area() - Bbox.overlap_area(this, other))



@dataclass
class Annotation:

    class_index: int # predicted class
    bbox: Bbox  # relative coordinates
    confidence: Optional[float]=None  # confidence score if any (no score for ground truth)

    def normalize(self, image_wh: Tuple[int, int]=(640, 360)) -> None:
        self.bbox = self.bbox.normalize(image_wh)

    def to_yolo_annotation_line(self) -> str:
        # dst format class xc yc w h confidence
        res = f"{int(self.class_index)} {self.bbox.xc} {self.bbox.yc} {self.bbox.w} {self.bbox.h}"
        if self.confidence is not None:
            res += f" {self.confidence}"
        return res
    
    def __hash__(self) -> int:
        return hash(id(self))
    
    def __eq__(self, other):
        return id(self) == id(other)



def read_validation_images(
        val_dir: str, 
        target_resize: tuple=(640, 360), 
        allowed_image_suffixes: tuple= ALLOWED_IMAGE_SUFFIXES
    ) -> List[Path]:
    candidates = [file for file in Path(val_dir).glob("*.*") if (file.is_file() and (file.suffix in allowed_image_suffixes))]
    return [file for file in candidates if has_valid_image_dimensions(file, target_resize)]


def has_valid_image_dimensions(
        image_path: Path,
        target_resize: tuple
    ) -> bool:
    """
    Check if resizing the image with the longer side as 640 while maintaining
    aspect ratio results in an image of size 640x360.

    :return: True if resizing results in 640x360, False otherwise.
    """
    width, height = Image.open(image_path).size
    # Calculate the multiple factor for width and height
    multiple_factor = width / target_resize[0] if width > height else height / target_resize[1]
    # Check if width is a multiple of 640 and height/multiple_factor is 360
    return width % target_resize[0] == 0 and int(height / multiple_factor) == target_resize[1]


def file_to_annotation(file_path: Union[str, Path]) -> List[Annotation]:
    if not Path(file_path).is_file():
        return []
    
    with open(file_path, 'r') as stream:
        file_data = stream.readlines()

    file_data = _pick_essential_strings(file_data)
    file_data = [_pick_essential_strings(item.split()) for item in file_data]
    file_data = [[int(item[0]), *map(float, item[1:])] for item in file_data]
    ret = []
    for annotation_line in file_data:
        ret.append(
            Annotation(
                annotation_line[0],
                Bbox(*annotation_line[1:5]),
                annotation_line[5] if len(annotation_line) == 6 else None
            )
        )
    return ret

def _str_essentials(item: str) -> str:
    return item.replace("\n", '').strip()

def _str_empty(item: str) -> bool:
    return _str_essentials(item) == ''

def _pick_essential_strings(iter: Iterable[str]) -> List:
    return [_str_essentials(item) for item in iter if not _str_empty(item)]


def load_stored_embeddings(
        folder_with_crops_and_embeddings: Union[str, Path],
        class_of_interest: int
    ) -> Tuple[List[Annotation], List[Path], List[Path]]:
    """
    @returns: tuple: annotations List[Annotation], embedding_files List[Path], image_paths List[Path]
    """

    image_paths = sorted(collect_crops_from_dir(folder_with_crops_and_embeddings))
    embedding_files = [item.with_suffix(".npy") for item in image_paths]
    annotations = [file_to_annotation(image_path.with_suffix(".txt"))[0] for image_path in image_paths] # 0 as one image one annotation


    # let's filter annotations by class of interest if there is one
    if class_of_interest is not None:
        condition = lambda item: item.class_index == class_of_interest
        filtered = [
            (a, e, i) for (a, e, i) in zip(annotations, embedding_files, image_paths)
            if condition(a)
        ]
        annotations, embedding_files, image_paths = map(list, zip(*filtered)) if filtered else ([], [], [])      

    if not all(embedding_file.is_file() for embedding_file in embedding_files):
        print(stylish_text("Embeddings are missing for some images. Seems dir is corrupted. Exiting.", TextStyles.FAIL))
        exit()
    
    return annotations, embedding_files, image_paths 


def collect_crops_from_dir(folder: Union[str, Path]) -> List[Path]:
    images = [
        image for image in Path(folder).glob("*.*") 
        if (image.is_file() and (image.suffix in ALLOWED_IMAGE_SUFFIXES))
    ]
    # condition = lambda w, h: ((w / 16) == (h / 9)) and (w % 16 == 0) 
    # images = [image for image in images if condition(*Image.open(image).size)]    
    return images     

def assign_detection_status_to_prediction(
        crop_annotation: Annotation, 
        ground_truth_annotation_file: Union[str, Path],
        tp_treshhold: float=0.4
    ) -> str:
    """
    is this cropped image TP, FP? (I do not assign FN as it is a status of ground truth markup. TN is not well-defined in detection problem) 
    """
    gt_annotations = file_to_annotation(ground_truth_annotation_file)
    for gt in gt_annotations:
        if gt.class_index != crop_annotation.class_index:
            continue
        if Bbox.IoU(gt.bbox, crop_annotation.bbox) >= tp_treshhold:
            return 'tp'
    return 'fp'


def load_ground_truth_markup_for_image_crop_file_paths(
        crop_file_paths: List[Path], 
        ground_truth_markup_dir: Union[str, Path]
    ) -> List[Path]:
    gt_files = [
        Path(ground_truth_markup_dir) /  f"{str(image_path.name).split('_cropped')[0]}.txt"
        for image_path in crop_file_paths
    ]  
    if not all(file.is_file() for file in gt_files):
        print(stylish_text("ground truth files not found for some croped images. Fix your directories", TextStyles.FAIL))
        exit()
    return gt_files

def any_suf_image_path(image_path_without_suf: Path) -> Path:
    for suf in ALLOWED_IMAGE_SUFFIXES:
        path_candidate = Path(f"{image_path_without_suf}{suf}")
        if path_candidate.is_file():
            return path_candidate
    return None


class TextStyles:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def stylish_text(line: str, style: Union[str, Sequence[str]]) -> str:
    """
    @param line: str line to colorize
    @param style: style from TextStyles class, i.e. TextStyles.HEADER or Sequence of TextStyles parameters 
                  to get bold underlined green text
    """
    if type(style) == str:
        return f"{style}{line}{TextStyles.ENDC}"
    else:
        has_get_item = hasattr(style, "__getitem__")
        has_len = hasattr(style, "__len__")
        if not (has_get_item and has_len):
            raise ValueError(f"text_color variable must be os Sequence type (to have len and getitem methods), but got {type(style)}")
        for separate_style in style:
            line = stylish_text(line, separate_style)
        return line
    

if __name__ == "__main__":
    bbox = Bbox(0.2, 0.3, 0.4, 0.35)
    breakpoint()