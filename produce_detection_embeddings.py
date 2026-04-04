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
from _yolov8_onnx_preprocessing import preprocess_source, preprocess
from concurrent.futures import ThreadPoolExecutor
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
    NETWORK_HEAD_OUTPUT_NAME = "output0"  # main YoloV8 output layer name

    def __init__(
        self,
        onnx_model_path: Union[str, Path],
        netron_layer_names: Optional[Sequence[str]] = None,
        output_alias_names: Optional[Dict[int, str]] = None,
        providers: List[str] = ["CUDAExecutionProvider"],
        imgsz: Optional[Sequence[int]] = None,
        save_crops=True,
        strategy: str = "default",
        embedding_tensors_hw_resolution_before_flattening: Optional[
            Sequence[int]
        ] = None,
        matryoshka_slices: int = 8,
    ) -> None:
        self.save_crops = save_crops
        self.onnx_model_path = onnx_model_path
        self.strategy = strategy
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
        # Number of equal "Matryoshka prefix" blocks to create when `strategy` is not "default"/"separate".
        # Each block contains 1/N of channels from EACH map, so any prefix of the final vector corresponds to
        # the same fraction of all maps (what `al_utils._slice_vector()` assumes).
        self.matryoshka_slices = int(matryoshka_slices)
        if embedding_tensors_hw_resolution_before_flattening is None:
            self.EMBEDDING_TENSORS_HW_RESOLUTION_BEFORE_FLATTENING = (
                (12, 6) if strategy == "default" else (3, 3)
            )
        else:
            if len(embedding_tensors_hw_resolution_before_flattening) != 2:
                raise ValueError(
                    "embedding_tensors_hw_resolution_before_flattening must be a 2-item (H, W) sequence"
                )
            h, w = embedding_tensors_hw_resolution_before_flattening
            self.EMBEDDING_TENSORS_HW_RESOLUTION_BEFORE_FLATTENING = (int(h), int(w))
        self.model = onnx.load(onnx_model_path)
        self.supports_batched_inference = False
        if self.model.graph.input:
            dims = self.model.graph.input[0].type.tensor_type.shape.dim
            if dims:
                first_dim = dims[0]
                self.supports_batched_inference = not (
                    first_dim.HasField("dim_value") and first_dim.dim_value == 1
                )
        self._extend_model_outputs_by_outputs_from_layers_of_interest()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.model.SerializeToString(),
            sess_options=sess_options,
            providers=self.providers,
        )
        # Stable order of layer output tensor names corresponding to self.netron_layer_names.
        # We need this because onnx session outputs are tensor names, and their order is not guaranteed
        # to match `netron_layer_names` directly.
        self._layer_output_tensor_names_in_order = (
            self._get_layer_output_tensor_names_in_netron_order()
        )

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
    ) -> Dict[Annotation, Union[np.ndarray, List[np.ndarray]]]:

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

        return self._compute_embeddings_from_outputs(image_inference_outputs, normalized_annotations)

    def _embedding_from_feature_vectors(
        self, feature_vectors: List[torch.Tensor]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        if self.strategy == "default":
            flattened_tensors = [t.flatten(start_dim=1) for t in feature_vectors]
            return torch.cat(flattened_tensors, dim=1).numpy()

        if self.strategy == "separate":
            return [t.flatten(start_dim=1).numpy() for t in feature_vectors]

        if len(feature_vectors) != 3:
            raise ValueError(
                f"Matryoshka embedding expects exactly 3 feature maps, got {len(feature_vectors)}. "
                f"Either pass exactly 3 `netron_layer_names` or use strategy='default'/'separate'."
            )

        sorted_feature_vectors = sorted(feature_vectors, key=lambda v: v.shape[1])
        t_low, t_mid, t_high = sorted_feature_vectors

        low_c = int(t_low.shape[1])
        mid_c = int(t_mid.shape[1])
        high_c = int(t_high.shape[1])

        preferred_slices = max(1, int(self.matryoshka_slices))
        slices = preferred_slices
        if (low_c % slices) or (mid_c % slices) or (high_c % slices):
            for cand in range(preferred_slices, 0, -1):
                if (low_c % cand == 0) and (mid_c % cand == 0) and (high_c % cand == 0):
                    slices = cand
                    break
        if (low_c % slices) or (mid_c % slices) or (high_c % slices):
            raise ValueError(
                "Could not find a common slice count to preserve Matryoshka prefix layout. "
                f"Got channels low={low_c}, mid={mid_c}, high={high_c}, preferred_slices={preferred_slices}. "
                "Tip: choose different 3 feature maps (netron layer names) or pass a smaller matryoshka_slices."
            )
        low_step = low_c // slices
        mid_step = mid_c // slices
        high_step = high_c // slices

        blocks = []
        for i in range(slices):
            low_part = t_low[:, i * low_step : (i + 1) * low_step, :, :]
            mid_part = t_mid[:, i * mid_step : (i + 1) * mid_step, :, :]
            high_part = t_high[:, i * high_step : (i + 1) * high_step, :, :]
            block = torch.cat([low_part, mid_part, high_part], dim=1)
            blocks.append(block.flatten(start_dim=1))

        return torch.cat(blocks, dim=1).numpy()

    def _compute_embeddings_from_outputs(
        self,
        image_inference_outputs: Dict[str, np.ndarray],
        normalized_annotations,
    ) -> Dict[Annotation, Union[np.ndarray, List[np.ndarray]]]:
        annotations = list(normalized_annotations)
        if not annotations:
            return {}

        output_size = self.EMBEDDING_TENSORS_HW_RESOLUTION_BEFORE_FLATTENING
        rois_per_layer: List[torch.Tensor] = []
        for output_tensor_name in self._layer_output_tensor_names_in_order:
            output_array = image_inference_outputs[output_tensor_name]
            input_tensor = torch.from_numpy(output_array).to(torch.float32)
            _, _, h, w = output_array.shape
            rois = torch.tensor(
                [
                    [0, w * ann.bbox.x_min, h * ann.bbox.y_min, w * ann.bbox.x_max, h * ann.bbox.y_max]
                    for ann in annotations
                ],
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            rois_per_layer.append(
                roi_align(
                    input_tensor,
                    rois,
                    output_size,
                    aligned=True,
                ).to(torch.float32)
            )

        annotation_embeddings_pairs = {}
        for idx, annotation in enumerate(annotations):
            feature_vectors = [layer_rois[idx : idx + 1] for layer_rois in rois_per_layer]
            annotation_embeddings_pairs[annotation] = self._embedding_from_feature_vectors(
                feature_vectors
            )

        return annotation_embeddings_pairs

    def _extend_model_outputs_by_outputs_from_layers_of_interest(self) -> None:
        """
        looping through model outputs will gives us only main model output defined by architecture
        If we want to get outputs from some hidden layers where it was not assumed initially
        we need to modify onnx file so that hidden layers of our interest would also produce outputs
        """
        # existing outputs
        org_outputs = [x.name for x in self.model.graph.output]

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

    def _get_layer_output_tensor_names_in_netron_order(self) -> List[str]:
        """
        @returns: list of output tensor names (strings) corresponding to nodes in `self.netron_layer_names`,
                  in the same order as `self.netron_layer_names`.
        """
        node_names_to_nodes = self._map_netron_layer_names_to_nodes()
        out_names = []
        for node_name in self.netron_layer_names:
            node = node_names_to_nodes[node_name]
            if not node.output:
                raise ValueError(f"Node '{node_name}' has no outputs")
            # Most relevant nodes have a single output tensor
            out_names.append(node.output[0])
        return out_names

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
        from_annotations_in_dir: bool = False,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        n: int = -1,
        random_images: bool = True,
        each_k_th_image: Optional[int] = None,
        batch_size: int = 8,
        io_workers: int = 4,
        save_crops: Optional[bool] = None,
    ):
        if save_crops is not None:
            self.save_crops = save_crops
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
            batch_size=batch_size,
            io_workers=io_workers,
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
        from_annotations_in_dir: bool = False,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        batch_size: int = 8,
        io_workers: int = 4,
    ):
        print(
            stylish_text(
                f"Running inference with model: {self.onnx_model_path}",
                style=TextStyles.OKBLUE,
            )
        )
        print(
            stylish_text(
                (
                    f"Batched inference (batch_size={batch_size}), threaded image loading"
                    if self.supports_batched_inference
                    else "Single-image inference (ONNX export is fixed to batch=1)"
                ),
                style=TextStyles.OKBLUE,
            )
        )

        def _read_image(path):
            return cv2.imread(str(path))

        embedding_and_crops_save_dir.mkdir(exist_ok=True, parents=True)
        effective_batch_size = batch_size if self.supports_batched_inference else 1
        pbar = tqdm(total=len(images))
        with ThreadPoolExecutor(max_workers=max(1, int(io_workers))) as read_pool:
            for batch_start in range(0, len(images), effective_batch_size):
                batch_paths = images[
                    batch_start : batch_start + effective_batch_size
                ]

                orig_imgs = list(read_pool.map(_read_image, batch_paths))
                preprocessed = preprocess(orig_imgs, self.imgsz)
                try:
                    batch_outputs = self._get_outputs_from_layers_of_interest(
                        preprocessed
                    )
                except ort.RuntimeException as exc:
                    if effective_batch_size == 1:
                        raise
                    if "cannot be reshaped" not in str(exc):
                        raise
                    print(
                        stylish_text(
                            "Falling back to batch_size=1 for this ONNX export",
                            style=TextStyles.WARNING,
                        )
                    )
                    self.supports_batched_inference = False
                    return self.produce_embeddings_for_images_from_list(
                        images,
                        embedding_and_crops_save_dir,
                        from_annotations_in_dir,
                        conf_thres,
                        iou_thres,
                        batch_size=1,
                    )

                for idx, image_path in enumerate(batch_paths):
                    per_img_outputs = {k: v[idx:idx+1] for k, v in batch_outputs.items()}

                    if from_annotations_in_dir:
                        normalized_annotations = file_to_annotation(image_path.with_suffix(".txt"))
                    else:
                        normalized_annotations = self.norm_pred_from_inference_outputs(
                            preprocessed[idx:idx+1], per_img_outputs,
                            [orig_imgs[idx]], conf_thres, iou_thres,
                        )

                    annotation_dict = self._compute_embeddings_from_outputs(
                        per_img_outputs, normalized_annotations,
                    )

                    for annotation, embedding in annotation_dict.items():
                        bbox = annotation.bbox
                        cropped_image_path = self._save_cropped_image(
                            image_path,
                            bbox,
                            embedding_and_crops_save_dir,
                            image_array=orig_imgs[idx],
                        )
                        if self.strategy == "separate":
                            for i, emb in enumerate(embedding):
                                np.save(cropped_image_path.with_suffix(f".m{i}.npy"), emb)
                        else:
                            np.save(cropped_image_path.with_suffix(".npy"), embedding)
                        Path(cropped_image_path.with_suffix(".txt")).write_text(
                            annotation.to_yolo_annotation_line()
                        )

                    pbar.update(1)
        pbar.close()

    def _next_cropped_output_path(self, image_path: Union[str, Path], output_dir: Path) -> Path:
        base_output_path = output_dir / (Path(image_path).stem + "_cropped")
        output_path = Path(str(base_output_path) + ".jpg")
        counter = 1
        while self._embedding_artifact_exists(output_path):
            output_path = base_output_path.with_name(
                base_output_path.name + f"_{counter}.jpg"
            )
            counter += 1
        return output_path

    def _embedding_artifact_exists(self, output_path: Path) -> bool:
        if output_path.is_file():
            return True
        sibling_paths = [
            output_path.with_suffix(".npy"),
            output_path.with_suffix(".txt"),
            output_path.with_suffix(".m0.npy"),
            output_path.with_suffix(".m1.npy"),
            output_path.with_suffix(".m2.npy"),
        ]
        return any(path.exists() for path in sibling_paths)

    def _save_cropped_image(
        self,
        image_path: Union[str, Path],
        bbox_obj: Bbox,
        output_dir: Path,
        image_array: Optional[np.ndarray] = None,
    ) -> Path:
        output_path = self._next_cropped_output_path(image_path, output_dir)

        if image_array is None:
            with Image.open(image_path) as img:
                image_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        img_h, img_w = image_array.shape[:2]
        abs_bbox = bbox_obj.to_absolute((img_w, img_h)).rectangle()
        for i in range(2):
            if abs_bbox[i] == abs_bbox[i + 2]:
                if abs_bbox[i]:
                    abs_bbox[i] -= 1
                else:
                    abs_bbox[i + 2] = 1

        if self.save_crops:
            x1, y1, x2, y2 = abs_bbox
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))
            cropped_img = image_array[y1:y2, x1:x2]
            cv2.imwrite(str(output_path), cropped_img)

        return output_path


if __name__ == "__main__":
    onnx_model_path = (
        "/home/setupishe/ultralytics/runs/detect/random_0.2/weights/best.onnx"
    )
    output_alias_names = {}

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
