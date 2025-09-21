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

# Optional HNSW backend
try:
    import hnswlib

    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False


def get_embedding_dim(npy_list):
    return np.load(npy_list[0]).squeeze(0).shape[0]


def _slice_vector(vec: np.ndarray, use_dim: int) -> np.ndarray:
    v = vec[:use_dim]
    n = np.linalg.norm(v) + 1e-12
    return v / n


def build_annoy_index(npy_list, embedding_dim, n_trees=10, use_dim=None):
    use_dim = embedding_dim if use_dim is None else int(use_dim)
    index = AnnoyIndex(use_dim, "angular")  # Use 'angular' for cosine-like distance
    idx = 0

    for file in tqdm(npy_list):
        emb = np.load(file).squeeze(0).astype(np.float32)
        emb = _slice_vector(emb, use_dim)
        index.add_item(idx, emb)
        idx += 1
    index.build(n_trees)
    return index


def build_hnsw_index(
    npy_list,
    embedding_dim,
    space="cosine",
    M=16,
    ef_construction=200,
    ef_search=50,
    use_dim=None,
):
    if not HAS_HNSWLIB:
        raise ImportError(
            "hnswlib is not installed. Please install it or use backend='annoy'."
        )
    use_dim = embedding_dim if use_dim is None else int(use_dim)
    index = hnswlib.Index(space=space, dim=use_dim)
    index.init_index(max_elements=len(npy_list), ef_construction=ef_construction, M=M)
    # Batch add for correct shape (num_items, dim)
    embeddings = []
    for file in tqdm(npy_list):
        emb = np.load(file).squeeze(0).astype(np.float32)
        emb = _slice_vector(emb, use_dim)
        embeddings.append(emb)
    data = np.vstack(embeddings).astype(np.float32)
    ids = np.arange(len(npy_list), dtype=np.int64)
    index.add_items(data, ids=ids)
    index.set_ef(ef_search)
    return index


def count_embeddings(folder):
    return sum(1 for filename in os.listdir(folder) if filename.endswith(".npy"))


def _score_from_distances(distances_list, mode: str) -> float:
    if mode == "distance":
        return float(distances_list[0])
    # density-like score (higher is more dense): inverse square sum
    return float(sum([1.0 / (d * d + 1e-6) for d in distances_list]))


def _collect_top_by_image(scored_files, top_n: int) -> list:
    # scored_files: List[Tuple[score, filepath]] already sorted descending or ascending per use
    res_files = []
    seen_imgs = set()
    for score, f in scored_files:
        name = os.path.basename(f)
        img_name = name[: name.index("_cropped")]
        if img_name in seen_imgs:
            continue
        seen_imgs.add(img_name)
        res_files.append(f)
        if len(res_files) >= top_n:
            break
    return res_files


def _load_matrix(npy_list, use_dim=None, normalize=True) -> np.ndarray:
    # Shape: (N, D)
    embs = []
    for file in npy_list:
        v = np.load(file).squeeze(0).astype(np.float32)
        if use_dim is not None:
            v = v[: int(use_dim)]
        if normalize:
            n = np.linalg.norm(v) + 1e-12
            v = v / n
        embs.append(v)
    return np.vstack(embs).astype(np.float32)


def _exact_nearest_distance_cosine(
    candidate_vec: np.ndarray, first_matrix: np.ndarray
) -> float:
    # first_matrix is L2-normalized; normalize candidate too
    v = candidate_vec.astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    sims = first_matrix @ v  # cosine similarities
    cos_dist = 1.0 - sims  # distances
    return float(np.min(cos_dist))


def select_embeddings(
    first_list,
    second_list,
    n=None,
    k=None,
    mode="distance",
    backend="annoy",
    coarse_to_fine=False,
):
    if mode not in ["distance", "density"]:
        raise ValueError("`mode` should be either in distance or density")
    if k is None and n is None:
        raise ValueError("either `n` of `k` should be specified")
    elif k is not None and n is not None:
        raise ValueError("`n` of `k` cannot both be specified")
    if backend not in ["annoy", "hnsw"]:
        raise ValueError("`backend` should be either 'annoy' or 'hnsw'")

    embedding_dim = get_embedding_dim(first_list)

    # Determine final k as integer count
    total_embeddings = len(second_list)
    k = int(total_embeddings // n) if n is not None else int(k)

    if not coarse_to_fine:
        if backend == "annoy":
            print("Building Annoy index from the first list...")
            index = build_annoy_index(first_list, embedding_dim, n_trees=10)
            print("Annoy index built.")
        else:
            print("Building HNSW index from the first list...")
            index = build_hnsw_index(first_list, embedding_dim)
            print("HNSW index built.")

        print(f"Selecting top {k} embeddings with the largest distances.")

        embeds_with_scores = []
        for file in tqdm(second_list):
            emb = np.load(file).squeeze(0).astype(np.float32)
            if backend == "annoy":
                nearest_idxs, distances = index.get_nns_by_vector(
                    emb, 1, include_distances=True
                )
                distances_list = distances
            else:
                labels, distances = index.knn_query(emb.reshape(1, -1), k=1)
                distances_list = distances[0].tolist()
            score = _score_from_distances(distances_list, mode)
            embeds_with_scores.append((score, file))
        reverse = mode == "distance"
        sorted_embeds = sorted(embeds_with_scores, key=lambda x: x[0], reverse=reverse)
        # Deduplicate by image and collect final list of image names
        res = set()
        for score, f in sorted_embeds:
            name = os.path.basename(f)
            img_name = name[: name.index("_cropped")]
            if len(res) < k:
                res.add(img_name)
            else:
                break
        return list(res)

    # Coarse-to-fine pipeline
    print("Coarse-to-fine selection enabled")
    k1 = min(len(second_list), max(1, k * 4))
    k2 = min(len(second_list), max(1, k * 2))

    # Stage 1: Annoy on 1/8 dims
    d1 = max(1, embedding_dim // 8)
    print(f"[Stage 1] Building Annoy index on {d1}/{embedding_dim} dims...")
    index1 = build_annoy_index(first_list, embedding_dim, n_trees=10, use_dim=d1)
    print("[Stage 1] Ranking candidates...")
    scores1 = []
    for file in tqdm(second_list):
        emb = np.load(file).squeeze(0).astype(np.float32)
        emb = _slice_vector(emb, d1)
        _, distances = index1.get_nns_by_vector(emb, 1, include_distances=True)
        score = _score_from_distances(distances, mode)
        scores1.append((score, file))
    reverse = mode == "distance"
    stage1_sorted = sorted(scores1, key=lambda x: x[0], reverse=reverse)
    candidates_stage1 = _collect_top_by_image(stage1_sorted, k1)

    # Stage 2: HNSW on 1/4 dims
    if not HAS_HNSWLIB:
        raise ImportError("hnswlib not installed; required for coarse-to-fine stage 2.")
    d2 = max(1, embedding_dim // 4)
    print(f"[Stage 2] Building HNSW index on {d2}/{embedding_dim} dims...")
    index2 = build_hnsw_index(first_list, embedding_dim, use_dim=d2)
    print("[Stage 2] Re-ranking candidates...")
    scores2 = []
    for file in tqdm(candidates_stage1):
        emb = np.load(file).squeeze(0).astype(np.float32)
        emb = _slice_vector(emb, d2)
        labels, distances = index2.knn_query(emb.reshape(1, -1), k=1)
        score = _score_from_distances(distances[0].tolist(), mode)
        scores2.append((score, file))
    stage2_sorted = sorted(scores2, key=lambda x: x[0], reverse=reverse)
    candidates_stage2 = _collect_top_by_image(stage2_sorted, k2)

    # Stage 3: Exact cosine KNN on full dims
    print(
        f"[Stage 3] Exact re-ranking on full vectors (dim = {embedding_dim}) (cosine distance)..."
    )
    first_matrix = _load_matrix(first_list, use_dim=None, normalize=True)
    scores3 = []
    for file in tqdm(candidates_stage2):
        emb = np.load(file).squeeze(0).astype(np.float32)
        min_dist = _exact_nearest_distance_cosine(emb, first_matrix)
        # For density mode, we can't do density with only exact 1-NN easily; keep using same scoring
        if mode == "distance":
            score = min_dist
        else:
            score = 1.0 / (min_dist * min_dist + 1e-6)
        scores3.append((score, file))
    stage3_sorted = sorted(scores3, key=lambda x: x[0], reverse=reverse)

    # Produce final list of image names
    res = set()
    for score, f in stage3_sorted:
        name = os.path.basename(f)
        img_name = name[: name.index("_cropped")]
        if len(res) < k:
            res.add(img_name)
        else:
            break
    return list(res)
