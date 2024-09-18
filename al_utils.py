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

def seg2bbox(filepath, to_path):
    with open(filepath, "r") as f:
        lines = [x.rstrip("\n") for x in f.readlines()]
    new_lines = []
    for line in lines:
        lst = line.split()
        new_line = [lst[0]]

        coords = [float(x) for x in lst[1:]]
        h, w = get_shape(txt2jpg(filepath))
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

        new_lines.append(
            " ".join([str(item) for item in [lst[0], xc, yc, width, height]]) + "\n"
        )
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

def select_embeddings(first_list, second_list, n=None, k=None):
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

    heap = []  # Min-heap to keep track of top-k largest distances

    for file in tqdm(second_list):
        emb = np.load(file).squeeze(0)

        # Find the nearest neighbor in the first folder
        nearest_idxs, distances = index.get_nns_by_vector(
            emb, 1, include_distances=True
        )
        min_distance = distances[0]

        if len(heap) < k: 
            heapq.heappush(heap, (min_distance, file))
        else:
            if min_distance > heap[0][0]: 
                heapq.heappushpop(heap, (min_distance, file)) 

    # Extract the embeddings from the heap
    selected_emb_paths = [item[1] for item in heap]
    
    return selected_emb_paths