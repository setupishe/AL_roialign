"""
Idea is to take feature maps from all backbone heads of YOLOv8 and try to locate what features
are corresponding to specific detection regions. Ofc we need to be sure that we are taking output of
hidden layers where spatial structure of output and input image are still align, i.e. leftmost element 
of feature map still corresponds to leftmost region of original image.
---------
Another workaround would be to make a script that runs two (or more) times:
one on full image and another one on image where only detected human is not blanked (or maybe a bit widened bbox)
and then try to see what activations change their output a lot and what are not: this is for the case if 
spatial alignment of features and input image is broken
"""

from collections import OrderedDict
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path, PosixPath
from typing import Union, Sequence, List, Dict, Tuple, Optional
import torch
from _yolov8_onnx_preprocessing import preprocess_source
from _yolov8_onnx_postprocessing import model_inference_to_annotations, postprocess
from utils import Annotation, TextStyles, stylish_text, Bbox, file_to_annotation
from PIL import Image
import math
from torchvision.ops import roi_align
from tqdm import tqdm
import shutil
import random
import cv2


class YoloEmbeddingsProducer:
    # Note: this is given as some class attribute but in fact cannot be easily changed here yet!
    # some preprocessing and postprocessing methods do depend on value 384 (i.e. on FullHD res of initial image) and should be rewritten
    YOLO_INPUT_SIZE: Tuple[int, int] = (640, 640)
    # scanning folders will collect only these extensions
    ALLOWED_IMAGE_SUFFIXES = ".jpeg", ".jpg", ".png"
    # all conv layers outputs will be resized (ROI Align) to this res to provide same output feature vector length
    # median bbox height in validation set is 0.23268499999999998 (relative units), median bbox width is 0.07114600000000004
    # if we take some layer with high enough resolution like /model.4/cv1/act/Mul which shape is 1, 192, 48, 80 where 48 (h) x 80 (w) is res
    # we get that corresponding res of median bbox will be 11.168879999999998 (h) x 5.691680000000003 (w)
    EMBEDDING_TENSORS_HW_RESOLUTION_BEFORE_FLATTENING = (12, 6)
    NETWORK_HEAD_OUTPUT_NAME = "output0"  # main YoloV8 output layer name

    def __init__(
        self,
        onnx_model_path: Union[str, Path],
        netron_layer_names: Optional[Sequence[str]] = None,
        output_alias_names: Optional[Dict[int, str]] = None,
        providers: List[str] = ["CUDAExecutionProvider"],
        imgsz: Optional[Sequence[int]] = None,
        save_crops=True,
    ) -> None:
        self.save_crops = save_crops
        self.onnx_model_path = onnx_model_path
        self.netron_layer_names = (
            netron_layer_names
            if netron_layer_names is not None
            else [
                "/model.15/cv2/act/Mul",
                "/model.22/Concat",
                "/model.22/Concat_1",
                "/model.22/Concat_2",
            ]
        )
        self.providers = providers

        self.model = onnx.load(onnx_model_path)
        self.session = ort.InferenceSession(
            self.model.SerializeToString(), providers=self.providers
        )  # use this session to modify model
        self._extend_model_outputs_by_outputs_from_layers_of_interest()  # add outputs from layers of interest
        self.session = ort.InferenceSession(
            self.model.SerializeToString(), providers=self.providers
        )  # reinitialize modified model

        self.output_alias_names = output_alias_names
        self.imgsz = imgsz if imgsz is not None else [640, 640]

    def predict(
        self,
        image_source: Union[str, Path, np.ndarray],
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
    ) -> List[Annotation]:
        orig_imgs = [
            (
                cv2.imread(str(image_source))
                if (isinstance(image_source, str) or isinstance(image_source, Path))
                else image_source
            )
        ]
        preprocessed_image, image_inference_outputs = self.get_image_inference_outputs(
            image_source
        )
        return self.norm_pred_from_inference_outputs(
            preprocessed_image,
            image_inference_outputs,
            orig_imgs,
            conf_thres,
            iou_thres,
        )

    def norm_pred_from_inference_outputs(
        self,
        preprocessed_image: np.ndarray,
        image_inference_outputs: Dict[str, np.ndarray],
        orig_imgs: List[np.ndarray],
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
    ) -> List[Annotation]:
        head_inference_outputs: np.ndarray = image_inference_outputs[
            YoloEmbeddingsProducer.NETWORK_HEAD_OUTPUT_NAME
        ]
        image_inference_outputs = torch.from_numpy(head_inference_outputs)
        unnormalized_predictions = postprocess(
            image_inference_outputs,
            preprocessed_image,
            orig_imgs,
            conf_thres,
            iou_thres,
        )[
            0
        ]  # [n, 6]
        prediction_annotations = model_inference_to_annotations(
            unnormalized_predictions, padding=0
        )
        for annotation in prediction_annotations:
            annotation.bbox = annotation.bbox.normalize(orig_imgs[0].shape[1::-1])
        return prediction_annotations

    def get_image_inference_outputs(
        self, image_source: Union[str, Path, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        preprocessed_image: np.ndarray = preprocess_source(image_source, self.imgsz)
        image_inference_outputs: Dict[str, np.ndarray] = (
            self._get_outputs_from_layers_of_interest(preprocessed_image)
        )
        return preprocessed_image, image_inference_outputs

    def get_image_embeddings(
        self,
        image_source: Union[str, Path, np.ndarray],
        load_annotations_from: Optional[
            Union[str, Path, List[Annotation]]
        ] = None,  # optional. Get them from predictions otherwise
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
    ) -> Dict[Annotation, np.ndarray]:

        # ==== get predicted bboxes, apply NMS and normalization if they came from predictions
        preprocessed_image, image_inference_outputs = self.get_image_inference_outputs(
            image_source
        )

        if load_annotations_from is None:
            orig_imgs = [
                (
                    cv2.imread(str(image_source))
                    if not isinstance(image_source, np.ndarray)
                    else image_source
                )
            ]
            normalized_annotations = self.norm_pred_from_inference_outputs(
                preprocessed_image,
                image_inference_outputs,
                orig_imgs,
                conf_thres,
                iou_thres,
            )
        else:
            if type(load_annotations_from) == list:
                if not all(
                    isinstance(item, Annotation) for item in load_annotations_from
                ):
                    raise TypeError(
                        (
                            "load_annotations_from have unexpected type. Expected: Optional[Union[str, Path, List[Annotation]]] "
                            f"got List[{type(load_annotations_from[0])}]"
                        )
                    )
                normalized_annotations = load_annotations_from
            elif isinstance(load_annotations_from, str) or isinstance(
                load_annotations_from, Path
            ):
                normalized_annotations = file_to_annotation(load_annotations_from)
            else:
                raise TypeError(
                    (
                        "load_annotations_from have unexpected type. Expected: Optional[Union[str, Path, List[Annotation]]] "
                        f"got {type(load_annotations_from)}"
                    )
                )

        # ==== get feature vectors from each hidden layer of interest that correspond to found bboxes
        annotation_embeddings_pairs = {}
        for annotation in normalized_annotations:
            feature_vectors = []
            bbox = annotation.bbox

            output_size = (
                YoloEmbeddingsProducer.EMBEDDING_TENSORS_HW_RESOLUTION_BEFORE_FLATTENING
            )
            for layer_name, output_array in image_inference_outputs.items():
                if (
                    layer_name == YoloEmbeddingsProducer.NETWORK_HEAD_OUTPUT_NAME
                ):  # no interest in output from yolo head
                    continue
                _, c, h, w = output_array.shape
                feature_map_box = [
                    w * bbox.x_min,
                    h * bbox.y_min,
                    w * bbox.x_max,
                    h * bbox.y_max,
                ]
                feature_vector = roi_align(
                    torch.tensor(output_array),
                    torch.tensor([0] + feature_map_box).unsqueeze(0),
                    output_size,
                    aligned=True,
                )
                feature_vectors.append(feature_vector.to(torch.float32))

            flattened_tensors = [t.flatten(start_dim=1) for t in feature_vectors]
            embedding_vector = torch.cat(flattened_tensors, dim=1).numpy()
            annotation_embeddings_pairs[annotation] = embedding_vector

        return annotation_embeddings_pairs

    def _extend_model_outputs_by_outputs_from_layers_of_interest(self) -> None:
        """
        looping through model outputs will gives us only main model output defined by architecture
        If we want to get outputs from some hidden layers where it was not assumed initially
        we need to modify onnx file so that hidden layers of our interest would also produce outputs
        """
        # existing outputs
        org_outputs = [x.name for x in self.session.get_outputs()]

        node_names_to_nodes = self._map_netron_layer_names_to_nodes()

        # adding outputs to hidden layers
        for node in self.model.graph.node:
            if (
                node not in node_names_to_nodes.values()
            ):  # exclude any node that does not correspond to a node with name in netron_layer_names
                continue
            for output in node.output:
                if output not in org_outputs:
                    self.model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    def _map_netron_layer_names_to_nodes(self) -> Dict[str, onnx.onnx_ml_pb2.NodeProto]:
        """
        @returns: mapping of node_name to node object
        @raises: ValueError if there are no nodes with names provided in netron_layer_names in the network
        """
        ret = {}
        for node in self.model.graph.node:
            for node_name in self.netron_layer_names:
                if node.name == node_name:
                    ret[node_name] = node
        not_found_names = set(self.netron_layer_names).difference(set(ret.keys()))
        if len(not_found_names):
            raise ValueError(
                f"NN nodes were not found for some provided netron_layer_names: {not_found_names}."
            )
        return ret

    def _get_outputs_from_layers_of_interest(
        self, input_array: np.ndarray
    ) -> Dict[str, np.ndarray]:
        outputs = [x.name for x in self.session.get_outputs()]
        ort_outs = self.session.run(
            outputs, {self.session.get_inputs()[0].name: input_array}
        )
        ## NOTE: uncomment here if you need to check output shape of a tensor from some layer
        # print(ort_outs[1].shape)  # ort_outs[0] is the head output. ort_outs[1] is the output of another layer (I suppose you use one extra)
        # print(f"height step = {384 / ort_outs[1].shape[2]} width step = {640 / ort_outs[1].shape[3]}") # (here is accuracy with this layer res)
        # breakpoint()
        ort_outs = OrderedDict(zip(outputs, ort_outs))
        return ort_outs

    def produce_embeddings_for_dir(
        self,
        dir_path: Union[str, Path],
        embedding_and_crops_save_dir: Union[str, Path],
        from_annotations_in_dir: bool = False,  # what is the source of annotations? exising markup or inference?
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        n: int = -1,  # -1 -> all images
        random_images: bool = True,  # matter only if n != -1
        each_k_th_image: Optional[int] = None,
    ):
        # inspecting if there are any files in temporal folder and whether to delete them or not
        embedding_and_crops_save_dir = Path(embedding_and_crops_save_dir)
        if not embedding_and_crops_save_dir.is_dir():
            embedding_and_crops_save_dir.mkdir(exist_ok=True, parents=True)
        else:
            if len(list(embedding_and_crops_save_dir.iterdir())):
                user_input: str = ""
                while user_input not in ("y", "n"):
                    print(
                        stylish_text(
                            f"tmp folder to store cropped images is not empty: {embedding_and_crops_save_dir}",
                            TextStyles.FAIL,
                        )
                    )
                    user_input = input("remove files? (y/n): ").lower().strip()
                if user_input == "y":
                    shutil.rmtree(embedding_and_crops_save_dir)
                    embedding_and_crops_save_dir.mkdir(exist_ok=True, parents=True)

        # collecting images for inference and embedding calculations
        images: List[Path] = self.__collect_images_from_dir(
            dir_path, full_hd_ratio=False
        )  # TODO: fix preprocessing in case image is not 16:9
        if each_k_th_image is None:
            if (random_images == True) and (n != -1):
                random.shuffle(images)
                images = images[:n]
            else:  # we take all images -> does not matter if they are suffled or not
                images = sorted(images)
        else:
            images = sorted(images)
            images = images[each_k_th_image - 1 :: each_k_th_image]

        return self.produce_embeddings_for_images_from_list(
            images,
            embedding_and_crops_save_dir,
            from_annotations_in_dir,
            conf_thres,
            iou_thres,
        )

    def __collect_images_from_dir(
        self, folder: Union[str, Path], full_hd_ratio: bool = False
    ) -> List[Path]:
        images = [
            image
            for image in Path(folder).glob("*.*")
            if (
                image.is_file()
                and (image.suffix in YoloEmbeddingsProducer.ALLOWED_IMAGE_SUFFIXES)
            )
        ]
        if full_hd_ratio == True:
            condition = lambda w, h: ((w / 16) == (h / 9)) and (w % 16 == 0)
            images = [image for image in images if condition(*Image.open(image).size)]
        return images

    def produce_embeddings_for_images_from_list(
        self,
        images: List[Path],
        embedding_and_crops_save_dir: Path,
        from_annotations_in_dir: bool = False,  # this allows us to get features for bboxes defined in the .txt annotation files in dir instead of by predictions
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
    ):
        print(
            stylish_text(
                f"Running inference with model: {self.onnx_model_path}",
                style=TextStyles.OKBLUE,
            )
        )
        print(
            stylish_text(
                "Getting predictions, image_cuts and calculating embeddings",
                style=TextStyles.OKBLUE,
            )
        )
        for image_path in tqdm(images):
            annotation_dict = (  # get embeddings from predictions if load_annotations_from_file is False else searh for filename as image_path.name with .txt
                self.get_image_embeddings(image_path, None, conf_thres, iou_thres)
                if not from_annotations_in_dir
                else self.get_image_embeddings(
                    image_path, load_annotations_from=image_path.with_suffix(".txt")
                )
            )

            for annotation, embedding in annotation_dict.items():
                bbox = annotation.bbox
                cropped_image_path = self._save_cropped_image(
                    image_path, bbox, embedding_and_crops_save_dir
                )
                np.save(cropped_image_path.with_suffix(".npy"), embedding)
                Path(cropped_image_path.with_suffix(".txt")).write_text(
                    annotation.to_yolo_annotation_line()
                )

    def _save_cropped_image(
        self, image_path: Union[str, Path], bbox_obj: Bbox, output_dir: Path
    ) -> Path:
        with Image.open(image_path) as img:
            # Convert relative bbox coordinates to absolute
            abs_bbox = bbox_obj.to_absolute((img.width, img.height)).rectangle()
            for i in range(2):
                if abs_bbox[i] == abs_bbox[i + 2]:
                    if abs_bbox[i]:
                        abs_bbox[i] -= 1
                    else:
                        abs_bbox[i + 2] = 1
            cropped_img = img.crop(abs_bbox)

            # Create a base output path without the counter
            base_output_path = output_dir / (Path(image_path).stem + "_cropped")
            output_path = Path(str(base_output_path) + ".jpg")
            counter = 1
            # Modify the output path with the counter if the file already exists
            while output_path.is_file():
                output_path = base_output_path.with_name(
                    base_output_path.name + f"_{counter}.jpg"
                )
                counter += 1
            if self.save_crops:
                cropped_img.save(output_path)
            return output_path


if __name__ == "__main__":
    onnx_model_path = (
        "/home/setupishe/ultralytics/runs/detect/random_0.2/weights/best.onnx"
    )
    output_alias_names = {
        "0": "person",
        "1": "bicycle",
        "2": "car",
        "3": "motorcycle",
        "4": "airplane",
        "5": "bus",
        "6": "train",
        "7": "truck",
        "8": "boat",
        "9": "traffic light",
        "10": "fire hydrant",
        "11": "stop sign",
        "12": "parking meter",
        "13": "bench",
        "14": "bird",
        "15": "cat",
        "16": "dog",
        "17": "horse",
        "18": "sheep",
        "19": "cow",
        "20": "elephant",
        "21": "bear",
        "22": "zebra",
        "23": "giraffe",
        "24": "backpack",
        "25": "umbrella",
        "26": "handbag",
        "27": "tie",
        "28": "suitcase",
        "29": "frisbee",
        "30": "skis",
        "31": "snowboard",
        "32": "sports ball",
        "33": "kite",
        "34": "baseball bat",
        "35": "baseball glove",
        "36": "skateboard",
        "37": "surfboard",
        "38": "tennis racket",
        "39": "bottle",
        "40": "wine glass",
        "41": "cup",
        "42": "fork",
        "43": "knife",
        "44": "spoon",
        "45": "bowl",
        "46": "banana",
        "47": "apple",
        "48": "sandwich",
        "49": "orange",
        "50": "broccoli",
        "51": "carrot",
        "52": "hot dog",
        "53": "pizza",
        "54": "donut",
        "55": "cake",
        "56": "chair",
        "57": "couch",
        "58": "potted plant",
        "59": "bed",
        "60": "dining table",
        "61": "toilet",
        "62": "tv",
        "63": "laptop",
        "64": "mouse",
        "65": "remote",
        "66": "keyboard",
        "67": "cell phone",
        "68": "microwave",
        "69": "oven",
        "70": "toaster",
        "71": "sink",
        "72": "refrigerator",
        "73": "book",
        "74": "clock",
        "75": "vase",
        "76": "scissors",
        "77": "teddy bear",
        "78": "hair drier",
        "79": "toothbrush",
    }

    # ========================================
    netron_layer_names = [
        # '/model.4/cv1/act/Mul' # step = 8  shape = (1, 192, 48, 80)
        "/model.22/Concat",  # (1, 66, 48, 80)
        "/model.22/Concat_1",  # (1, 66, 24, 40)
        "/model.22/Concat_2",  # (1, 66, 12, 20)
        # '/model.22/Concat_6'
    ]
    yep = YoloEmbeddingsProducer(
        onnx_model_path, netron_layer_names, output_alias_names  # , save_crops=False
    )
    yep.produce_embeddings_for_dir(
        dir_path="/home/setupishe/bel_conf/remainder_imgs_0.2",
        embedding_and_crops_save_dir="/home/setupishe/bel_conf/remainder_embeds_0.2",
        from_annotations_in_dir=True,
        conf_thres=0.6,
        iou_thres=0.4,
        n=-1,
        random_images=False,
        each_k_th_image=None,
    )
