import sys
from pathlib import Path, PosixPath
from PIL import Image
import numpy as np
from typing import Sequence, Union, Tuple, List
import cv2
import torch

TARGET_SIZE = 640
TARGET_IMAGE_SIZE = 640, 384
BACKGROUND_FILL_COLOR = 114, 114, 114

# def preprocess_PIL_image(
#         img: Union[str, Path, Image.Image], 
#         target_size: int=TARGET_SIZE, 
#         bg_fill: Tuple[int, int, int]=BACKGROUND_FILL_COLOR, 
#         no_padding_preprocess: bool=True
#     ) -> np.ndarray:
    
#     if isinstance(img, str) or isinstance(img, PosixPath):
#         image = Image.open(img)
#     elif isinstance(img, Image.Image):
#         image = img
#     else:
#         raise ValueError(f"Unsupported argument type for 'img'. Expected Union[Path, Image.Image], got {type(img)}")
    
#     image = resize_and_paste_on_grey_background(image, target_size, bg_fill, no_padding_preprocess)
#     return image_to_array(image)


# def resize_and_paste_on_grey_background(
#         image: Image.Image, 
#         target_size: int, 
#         bg_fill: Tuple[int, int, int],
#         no_padding_preprocess: bool=False
#     ) -> Image.Image:
#     """
#     the yolov8 image resizing algorithm is:
#     1) resize max side to 640 and lower side accordingly
#     2) if lower side is not a multiple of 32 we prolong it to the nearest multiple of 32 which is greater than lower side
#        (thus we will not loose image data)
#     3) we paste image from 1) to the center of 2)
#     """
#     if (max(image.size) != target_size) or (min(image.size) % 32 != 0):  # means resize is needed
#         resize_coeff = target_size / max(image.size)
#         new_size = np.array(image.size) * resize_coeff
#         new_size = np.rint(np.minimum(new_size, target_size)).astype(np.int32)
        
#         background_image_size = [s for s in (new_size)]
#         lower_side_index = np.argmin(new_size)
#         max_side_index = int(not lower_side_index)
#         lower_side = new_size[lower_side_index]
#         if lower_side % 32 != 0:
#             lower_side = int(np.ceil(lower_side / 32)) * 32
#             background_image_size[lower_side_index] = lower_side
#             background_image_size[max_side_index] = target_size
#             background_image_size = tuple(background_image_size)
        
#         if no_padding_preprocess:
#             image = image.resize(background_image_size ,Image.Resampling.LANCZOS)
#             return image
#         else:
#             image = image.resize(new_size, Image.Resampling.LANCZOS)    

#         background_image = Image.new("RGB", background_image_size, bg_fill)

#         # Calculate the position to paste the resized image (centered)
#         top_left_x = (background_image.width - image.width) // 2
#         top_left_y = (background_image.height - image.height) // 2

#         # Paste the resized image onto the new image
#         background_image.paste(image, (top_left_x, top_left_y))
#         image = background_image
#     return image


# def image_to_array(image: Image.Image) -> np.ndarray:
#     """
#     here images are already of size 640 x 384 obtained by pasting 640 x 360 images into 640 x 384 (114, 114, 114) array
#     """
#     img_data = np.array(image) / 255.0
#     img_data = np.transpose(img_data, axes=[2, 0, 1]).astype(np.float32)
#     return img_data[np.newaxis, ...]


# def core_like_preprocessing(image_source: Union[str, Path, np.ndarray]):
#     image = _resize_and_pad_image(image_source)
#     return _normalize_and_transpose_image(image)


# def _resize_and_pad_image(image_source: Union[str, Path, np.ndarray]):
#     """
#     Resize the image preserving the aspect ratio and pad it to match the target dimensions using OpenCV.
#     :param image_path: Path to the input image.
#     :return: Resized and padded image.
#     """
#     target_width, target_height = TARGET_IMAGE_SIZE

#     # Read the image
#     if type(image_source) in (str, PosixPath):
#         img = cv2.imread(str(image_source))
#     elif type(image_source) == np.ndarray:
#         img = image_source
#     else:
#         raise TypeError(
#             f"Incorrect argument type for image_source in function: _resize_and_pad_image. Expected Union[str, Path, np.ndarray], received {type(image_source)}"
#         )

#     # Calculate scales and choose the smaller scale to preserve aspect ratio
#     width_scale = target_width / img.shape[1]
#     height_scale = target_height / img.shape[0]
#     scale = min(width_scale, height_scale)

#     # Calculate new dimensions
#     resized_width = int(scale * img.shape[1])
#     resized_height = int(scale * img.shape[0])

#     # Resize the image
#     resized_img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

#     # Calculate padding
#     delta_w = target_width - resized_width
#     delta_h = target_height - resized_height
#     padding_top = delta_h // 2
#     padding_bottom = delta_h - padding_top
#     padding_left = delta_w // 2
#     padding_right = delta_w - padding_left

#     # Apply padding
#     padded_img = cv2.copyMakeBorder(
#         resized_img, padding_top, padding_bottom, padding_left, padding_right, 
#         cv2.BORDER_CONSTANT, value=BACKGROUND_FILL_COLOR
#     )
#     # cv2.imwrite("/home/devel/NVI/personal_scripts/int8Calibrator/debug/padded_img.jpg", padded_img)
#     # exit()
#     return padded_img


# def _normalize_and_transpose_image(image: np.ndarray) -> np.ndarray:
#     img_data = np.array(image) / 255.0
#     img_data = np.transpose(img_data, axes=[2, 0, 1]).astype(np.float32)
#     return img_data[np.newaxis, ...]


##### //////////////////////////////////////////////////////////////////////
##### //////////////////////////////////////////////////////////////////////
##### //////////////////////////////////////////////////////////////////////
##### ////// YOLO PREPROCESSING TAKEN DIRECTLY FROM ULTRALITICS REPO ///////
##### //////////////////////////////////////////////////////////////////////
##### //////////////////////////////////////////////////////////////////////
##### //////////////////////////////////////////////////////////////////////

def preprocess_source(
        src: Union[str, Path, np.ndarray, List[Union[str, Path]]],
        imgsz: Sequence[int]
    ) -> np.ndarray:
    """
    @returns: np.ndarray (B, C, H, W), i.e. (1, 3, 384, 640)
    """
    if isinstance(src, Path) or isinstance(src, str):
        source: List[np.ndarray] = [cv2.imread(str(src))]
    elif isinstance(src, np.ndarray):
        source: List[np.ndarray] = [src]
    elif isinstance(src, list) and all((isinstance(item, str) or isinstance(item, Path)) for item in src):
        source: List[np.ndarray] = [cv2.imread(str(item)) for item in src]
    else:
        raise TypeError(f"unknown data type for src variable. Expected Union[str, Path, np.ndarray, List[Union[str, Path]]] got {type(src)}")
    return preprocess(source, imgsz)

def preprocess(im: List[np.ndarray], imgsz: Sequence[int]) -> np.ndarray:
    """
    My Note: in Ultralitics framework what is coming into preprocess function is a list of np.ndarray image representation
             taken from batch. There is a bit of clarification below but I will be fixing it to List[np.ndarray]

    Prepares input image before inference.
    Args:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    assert (type(im) == list) and all(type(item) == np.ndarray for item in im) 
    pretransformed_batch = pre_transform_batch(im, imgsz)
    im = np.stack(pretransformed_batch)
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # im[..., ::-1] is BGR to RGB, transpose((0, 3, 1, 2) is BHWC to BCHW, (n, 3, h, w) i.e. (1, 3, 384, 640)
    im = np.ascontiguousarray(im)  # contiguous
    im = im.astype(np.float32)
    im /= 255  # 0 - 255 to 0.0 - 1.0
    return im

def pre_transform_batch(im: List[np.ndarray], imgsz: Sequence[int]):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """
    return [pre_transform_image(img=x, imgsz=imgsz) for x in im]

def pre_transform_image(img: np.ndarray, imgsz: Sequence[int]) -> np.ndarray:
    shape = img.shape[:2] # current shape [height, width]
    new_shape = imgsz

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) # in case of FullHD to 384, 640: min(384 / 1080 ~ 0.35, 640 / 1920 ~ 0.33) = 0.33(3)
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # in case of FullHD: = (640, 360)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding  (in case of FullHD: 0, 24)
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize (i.e. in case of FullHD: (1920, 1080) != (640, 360))
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    return img

if __name__ == "__main__":
    image = "/home/devel/NVI/personal_scripts/int8Calibrator/Coarse_NVI_office_NVI_office_multicamera_oleg_devices_walking_201_oleg_devices_walking.mp4.marked_frames_000351rc.jpg"
    preprocess_PIL_image(image)