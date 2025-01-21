import random
import os, glob, shutil
from PIL import Image
import cv2
import pickle


def pickle_save(filepath, obj):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def get_shape(img_path):
    with Image.open(img_path) as img:
        width, height = img.size
    return height, width


from tqdm import tqdm


def force_mkdir(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def get_shape(img_path):
    with Image.open(img_path) as img:
        width, height = img.size
    return height, width


def txt2jpg(inp):
    return inp.replace("/labels", "/images").replace(".txt", ".jpg")


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


def segfile2bboxfile(filepath, to_path, seg2line=True):
    with open(filepath, "r") as f:
        lines = [x.rstrip("\n") for x in f.readlines()]
    new_lines = []
    for line in lines:
        if seg2line:
            line = segline2bboxline(line, get_shape(txt2jpg(filepath)))

        new_lines.append(line + "\n")
    with open(to_path, "w") as f:
        f.writelines(new_lines)


import numpy as np
from annoy import AnnoyIndex
import heapq


def get_embedding_dim(npy_list):
    return np.load(npy_list[0]).squeeze(0).shape[0]


def build_annoy_index(npy_list, embedding_dim, n_trees=10):
    index = AnnoyIndex(embedding_dim, "angular")  # Use 'angular' for cosine similarity
    idx = 0

    for file in tqdm(npy_list):
        emb = np.load(file).squeeze(0)
        index.add_item(idx, emb)
        idx += 1
    index.build(n_trees)
    return index


def count_embeddings(folder):
    return sum(1 for filename in os.listdir(folder) if filename.endswith(".npy"))


def select_embeddings(first_list, second_list, n=None, k=None, mode="distance"):
    if mode not in ["distance", "density"]:
        raise ValueError("`mode` should be either in distance or density")
    if k is None and n is None:
        raise ValueError("either `n` of `k` should be specified")
    elif k is not None and n is not None:
        raise ValueError("`n` of `k` cannot both be specified")

    embedding_dim = get_embedding_dim(first_list)

    print("Building Annoy index from the first list...")
    index = build_annoy_index(first_list, embedding_dim, n_trees=10)
    print("Annoy index built.")

    total_embeddings = len(second_list)

    k = total_embeddings // n if n is not None else k
    print(f"Selecting top {k} embeddings with the largest distances.")

    embeds_with_distances = []  # Min-heap to keep track of top-k largest distances

    for file in tqdm(second_list):
        emb = np.load(file).squeeze(0)

        # Find the nearest neighbor in the first folder
        nearest_idxs, distances = index.get_nns_by_vector(
            emb, 1, include_distances=True
        )
        if mode == "distance":
            min_distance = distances[0]
        else:
            min_distance = sum([1 / (d**2 + 1e-6) for d in distances])

        embeds_with_distances.append((min_distance, file))
    # Extract the embeddings from the heap
    sorted_embeds = sorted(embeds_with_distances, key=lambda x: x[0], reverse=True)

    res = set()
    for pair in sorted_embeds:
        name = os.path.basename(pair[1]).split("_")[0]
        if len(res) < k:
            res.add(name)
        else:
            break

    return list(res)
