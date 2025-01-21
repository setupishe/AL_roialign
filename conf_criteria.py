import os, glob, shutil
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--from-fraction")
parser.add_argument("--to-fraction")
parser.add_argument("--from-split")
parser.add_argument("--dataset-name")
parser.add_argument("--bg2all-ratio")
parser.add_argument("--default-split")
parser.add_argument("--weights")
parser.add_argument("--split-name")
parser.add_argument("--cleanup", action="store_true")  # on/off flag
parser.add_argument("--seg2line", action="store_true")  # on/off flag

args = parser.parse_args()

from_fraction = float(args.from_fraction)
to_fraction = float(args.to_fraction)
from_split = args.from_split
dataset_name = args.dataset_name
default_split = args.default_split
weights = args.weights
bg2all_ratio = float(args.bg2all_ratio)
split_name = args.split_name
cleanup = args.cleanup
seg2line = args.seg2line

# # CLI arguments

# from_fraction = 0.6
# to_fraction = 0.7
# dataset_name = "coco"
# from_split = "train2017_0.6_confidences.txt"
# split_name = "confidences"

# iou_threshold = 0.7  # ultralytics default
# conf = 0.278  # get from from_split fscore chart

# weights = "/home/setupishe/ultralytics/runs/detect/confidences_0.6/weights/best.pt"

# cleanup = True

iou_threshold = 0.7  # ultralytics default

dataset_folder = f"/home/setupishe/datasets/{dataset_name}"
to_split = f"train_{to_fraction}_{split_name}.txt"
from_split = os.path.join(dataset_folder, from_split)
to_split = os.path.join(dataset_folder, to_split)
default_split = os.path.join(dataset_folder, default_split)

conf_path = "/".join(weights.split("/")[:-2]) + "/best_conf.txt"
with open(conf_path, "r") as f:
    conf = float(f.readline())


def txt2filelist(filepath):
    with open(filepath, "r") as f:
        return [x.rstrip("\n") for x in f.readlines()]


from_split_lst = txt2filelist(from_split)
default_split_lst = txt2filelist(default_split)

num_total = len(default_split_lst) * (to_fraction - from_fraction)
bg_num = int(num_total * bg2all_ratio)
frg_num = int(num_total - bg_num)

inference_list = list(set(default_split_lst) - set(from_split_lst))

inference_filepath = "inference_list.txt"
with open(inference_filepath, "w") as f:
    f.writelines([dataset_folder + item[1:] + "\n" for item in inference_list])

inference_name = "inference_results"
preds_folder = f"/home/setupishe/ultralytics/runs/detect/{inference_name}/labels"

cmd = f"yolo predict model={weights} source='{inference_filepath}' conf={conf} iou={iou_threshold} name={inference_name} save=False save_conf=True save_txt=True batch=64"


def prettyprint(msg):
    print("=" * 80)
    print("|   " + msg)
    print("=" * 80)
    print("\n")


prettyprint("Infering model...")
if not os.path.exists(preds_folder):
    os.system(cmd)


def segline2bboxline(line, shape):
    lst = line.split()

    coords = [float(x) for x in lst[1:]]
    h, w = shape
    first = [x for i, x in enumerate(coords) if i % 2 == 0]
    second = [x for i, x in enumerate(coords) if i % 2 == 1]
    for i in range(len(first)):
        first[i] = int(first[i] * w)
        second[i] = int(second[i] * h)

    xmin = min(first)
    xmax = max(first)
    ymin = min(second)
    ymax = max(second)

    width = (xmax - xmin) / w
    height = (ymax - ymin) / h
    xc = (xmax + xmin) / w / 2
    yc = (ymax + ymin) / h / 2

    res = " ".join([str(item) for item in [lst[0], xc, yc, width, height]])

    return res


from dataclasses import dataclass
from typing import List


def yolo2standard(image_shape, yolo_bbox):
    """
    Converts a bounding box in YOLO format to traditional format.

    :param image_shape: shape of the image as (height, width)
    :param yolo_bbox: bounding box in YOLO format as (center_x, center_y, width, height)
    :return: bounding box in traditional format as (x_min, y_min, x_max, y_max)
    """
    # Extract image dimensions
    img_height, img_width = image_shape[:2]

    # Extract the YOLO bounding box components
    center_x, center_y, width, height = yolo_bbox

    # Convert to traditional bounding box format
    x_min = int((center_x - width / 2.0) * img_width)
    y_min = int((center_y - height / 2.0) * img_height)
    x_max = int((center_x + width / 2.0) * img_width)
    y_max = int((center_y + height / 2.0) * img_height)
    return (x_min, y_min, x_max, y_max)


@dataclass
class Annotation:
    bbox: List[float]
    conf: float = None
    cls: int = None

    def __eq__(self, other):
        if not isinstance(other, Annotation):
            return NotImplemented
        return (
            self.bbox == other.bbox
            and self.conf == other.conf
            and self.cls == other.cls
        )


def txt2anno(txt, shape=None, verbose=True, seg2line=False, absolute_bbox=False):
    if seg2line and shape is None:
        raise ValueError("`seg2line=True` requires `shape` to be specified")
    if absolute_bbox and shape is None:
        raise ValueError("`absolute_bbox=True` requires `shape` to be specified")
    res = {}
    if os.path.exists(txt):
        with open(txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip("\n")
                if seg2line:
                    line = segline2bboxline(line, shape=shape)
                lst = line.split()
                bbox = [float(x) for x in lst[1:5]]
                conf = float(lst[5]) if len(lst) == 6 else None
                cls = int(lst[0])

                if cls not in res:
                    res[cls] = []
                if absolute_bbox:
                    H, W = shape
                    bbox = yolo2standard((H, W), bbox)
                res[cls].append(Annotation(bbox, conf, cls))
    elif verbose:
        print(f"anno file {txt} does not exist")
    return res


from al_utils import *


def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def jpg2txt(inp):
    return inp.replace("/images", "/labels").replace(".jpg", ".txt")


def txt2jpg(inp):
    return inp.replace("/labels", "/images").replace(".txt", ".jpg")


@dataclass
class Sample:

    filepath: str = None
    fscore: int = None
    status: str = None


samples = []
from tqdm import tqdm


def fscore(tp, fp, fn, eps=1e-6):
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


preds_folder
for img_path in tqdm(inference_list):
    sample = Sample()
    sample.filepath = img_path
    name = os.path.basename(img_path)

    pred_path = os.path.join(preds_folder, jpg2txt(name))
    label_path = jpg2txt(dataset_folder + img_path[1:])

    if not os.path.exists(pred_path):
        os.mknod(pred_path)

    tp = fp = fn = 0

    preds_anno = txt2anno(pred_path)
    labels_anno = txt2anno(
        label_path, seg2line=seg2line, shape=get_shape(txt2jpg(label_path))
    )
    if len(labels_anno) == 0:
        sample.status = "bg"
        fp += sum([x.conf for key in preds_anno for x in preds_anno[key]])
    else:

        for key in preds_anno:
            if key in labels_anno:
                for pred in preds_anno[key]:
                    found = False
                    for label in labels_anno[key]:
                        if bb_iou(pred.bbox, label.bbox) > iou_threshold:
                            tp += pred.conf
                            labels_anno[key].remove(label)
                            found = True
                            break
                    if not found:
                        fp += pred.conf
            else:
                fp += sum([x.conf for x in preds_anno[key]])
        fn += sum([1 for key in labels_anno for x in labels_anno[key]])

    sample.fscore = fscore(tp, fp, fn)
    samples.append(sample)

prettyprint("Collecting stats...")
sorted_samples = sorted(samples, key=lambda x: x.fscore)
import matplotlib.pyplot as plt

fscores = [x.fscore for x in sorted_samples]
bgs, frgs = [], []

for sample in sorted_samples:
    if sample.status == "bg":
        if len(bgs) < bg_num:
            bgs.append(sample.filepath)
    elif len(frgs) < frg_num:
        frgs.append(sample.filepath)

final_list = from_split_lst + bgs + frgs
with open(to_split, "w") as f:
    f.writelines([x + "\n" for x in final_list])

to_yaml = f"{dataset_name}_{to_fraction}_{split_name}.yaml"
yaml_folder = f"/home/setupishe/ultralytics/ultralytics/cfg/datasets"
original_yaml = f"{dataset_name}.yaml"


with open(os.path.join(yaml_folder, original_yaml), "r") as from_file:
    lines = from_file.readlines()

for i, line in enumerate(lines):
    if "train: train2017.txt" in line:
        lines[i] = (
            lines[i]
            .replace("train2017.txt", os.path.basename(to_split))
            .replace("118287", str(len(final_list)))
        )
with open(
    os.path.join(yaml_folder, to_yaml),
    "w",
) as to_file:
    to_file.writelines(lines)

prettyprint(f"`{os.path.join(yaml_folder, to_yaml)}` saved successfully.")

if cleanup:
    shutil.rmtree(preds_folder.rstrip("labels"))
    os.remove("inference_list.txt")
