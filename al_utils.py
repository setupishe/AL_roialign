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


def _exact_knn_distances_cosine(
    candidate_vec: np.ndarray, first_matrix: np.ndarray, k: int
) -> list:
    """
    Return the k smallest cosine distances (1 - cosine similarity) between candidate_vec and rows of first_matrix.
    first_matrix is assumed L2-normalized.
    """
    v = candidate_vec.astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    sims = first_matrix @ v
    cos_dist = 1.0 - sims
    k = int(max(1, min(k, cos_dist.shape[0])))
    # Use partial selection for efficiency
    smallest = np.partition(cos_dist, k - 1)[:k]
    return smallest.astype(np.float32).tolist()


def _select_by_granularity_variance(
    first_list,
    second_list,
    k,
    granularity_divs=None,
):
    """
    Score each candidate in second_list by the variance of its 1-NN cosine distance
    to the labeled pool (first_list) across multiple embedding granularity levels.

    Granularity levels are defined as prefixes of the full embedding dimension:
      d_g = embedding_dim // div  for each div in granularity_divs

    High variance means the example's proximity to the labeled pool changes
    significantly across scales — a signal of ambiguity or boundary proximity.
    """
    if granularity_divs is None:
        granularity_divs = [8, 4, 2, 1]

    embedding_dim = get_embedding_dim(first_list)
    granularity_dims = [max(1, embedding_dim // int(d)) for d in granularity_divs]
    # Deduplicate while preserving order (coarsest → finest)
    seen_dims = set()
    unique_dims = []
    for d in granularity_dims:
        if d not in seen_dims:
            seen_dims.add(d)
            unique_dims.append(d)
    granularity_dims = unique_dims

    print(
        f"[matryoshka_variance] granularity dims: {granularity_dims} "
        f"(from embedding_dim={embedding_dim})"
    )

    # Pre-load labeled pool matrix at each granularity level
    print("[matryoshka_variance] Pre-loading labeled pool matrices...")
    first_matrices = []
    for dim in granularity_dims:
        mat = _load_matrix(first_list, use_dim=dim, normalize=True)
        first_matrices.append(mat)

    # Score each candidate
    scored = []
    for file in tqdm(second_list, desc="Scoring candidates (granularity variance)"):
        vec = np.load(file).squeeze(0).astype(np.float32)
        dists = []
        for mat, dim in zip(first_matrices, granularity_dims):
            v = _slice_vector(vec, dim)
            cos_dists = 1.0 - (mat @ v)
            dists.append(float(np.min(cos_dists)))
        score = float(np.var(dists))
        scored.append((score, file))

    # Sort descending — highest variance first
    scored.sort(key=lambda x: x[0], reverse=True)

    res = []
    seen = set()
    for score, f in scored:
        name = os.path.basename(f)
        img_name = name[: name.index("_cropped")]
        if img_name in seen:
            continue
        seen.add(img_name)
        res.append(img_name)
        if len(res) >= k:
            break
    return res



def _select_by_gbm_oracle(
    first_list, second_list, k, from_fraction, to_fraction,
    gbm_model_path="gbm_oracle_balanced.pkl",
):
    """
    Select images using a pre-trained GBM that predicts oracle utility
    from embedding-derived proxy features + fraction context.
    """
    import pickle, json

    with open(gbm_model_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    feat_names = saved["feat_names"]
    use_dims = saved["use_dims"]

    embedding_dim = get_embedding_dim(first_list)
    total_train_imgs = 16551  # VOC hardcoded for now

    pool_size = len(set(
        os.path.basename(f)[:os.path.basename(f).index("_cropped")]
        for f in first_list
    ))

    # Load all embeddings into memory
    print(f"[GBM] Loading {len(first_list)} labeled + {len(second_list)} unlabeled embeddings...")
    L_vecs = np.array([np.load(f).squeeze(0).astype(np.float32) for f in first_list])
    U_vecs = np.array([np.load(f).squeeze(0).astype(np.float32) for f in second_list])
    U_stems = [os.path.basename(f)[:os.path.basename(f).index("_cropped")] for f in second_list]

    # 1-NN at each prefix dim
    nn_dists = {}
    for d in use_dims:
        d = min(d, embedding_dim)
        Ln = L_vecs[:, :d].copy()
        Ln /= np.maximum(np.linalg.norm(Ln, axis=1, keepdims=True), 1e-12)
        dists = np.empty(U_vecs.shape[0], dtype=np.float32)
        for s in range(0, U_vecs.shape[0], 1000):
            e = min(s + 1000, U_vecs.shape[0])
            Un = U_vecs[s:e, :d].copy()
            Un /= np.maximum(np.linalg.norm(Un, axis=1, keepdims=True), 1e-12)
            sim = Un @ Ln.T
            dists[s:e] = 1.0 - np.max(sim, axis=1)
        nn_dists[d] = dists
        print(f"[GBM]   dim={d}: mean_dist={np.mean(dists):.4f}")

    # Per-image features
    from collections import defaultdict
    by_img = defaultdict(list)
    for i, stem in enumerate(U_stems):
        by_img[stem].append(i)

    img_features = {}
    for stem, idxs in by_img.items():
        feat = [len(idxs), from_fraction, to_fraction, pool_size / total_train_imgs]
        for d in use_dims:
            d = min(d, embedding_dim)
            ds = nn_dists[d][idxs]
            feat.extend([float(np.max(ds)), float(np.mean(ds)), float(np.sum(ds)),
                         float(np.std(ds)) if len(ds) > 1 else 0.0])
        pcv = np.var(np.stack([nn_dists[min(d, embedding_dim)][idxs] for d in use_dims], axis=0), axis=0)
        feat.extend([float(np.mean(pcv)), float(np.max(pcv)), float(np.sum(pcv))])
        img_features[stem] = feat

    # Predict oracle scores
    imgs = sorted(img_features.keys())
    X = np.array([img_features[img] for img in imgs])
    scores = model.predict(X)

    # Rank and select top-k
    ranked = sorted(zip(scores, imgs), key=lambda x: x[0], reverse=True)
    selected = [img for _, img in ranked[:k]]
    print(f"[GBM] Selected {len(selected)} images (score range: {ranked[0][0]:.2f} to {ranked[-1][0]:.2f})")
    return selected



def _build_class_weights_from_pool(first_list, epsilon=0.01):
    """
    Compute per-class inverse-frequency weights from labeled pool crop .txt files.
    Each .txt has format: class xc yc w h [conf]
    Returns dict: class_id -> weight (higher = rarer in pool).
    """
    from collections import Counter
    class_counts = Counter()
    for f in first_list:
        txt = f.replace(".npy", ".txt")
        if not os.path.exists(txt):
            continue
        with open(txt) as fh:
            line = fh.readline().strip()
            if line:
                cls = int(line.split()[0])
                class_counts[cls] += 1
    total = max(1, sum(class_counts.values()))
    weights = {cls: 1.0 / (cnt / total + epsilon) for cls, cnt in class_counts.items()}
    default_weight = 1.0 / epsilon  # unseen classes get max weight
    return weights, default_weight


def _aggregate_scores_by_image(scored_files, k, aggregation="max", class_weights=None, default_class_weight=1.0):
    """
    Group per-crop scores by image and produce a ranked list of image names.
    aggregation: "max" (legacy), "sum", "mean", "crop_weighted" (mean*sqrt(n)),
                 "class_weighted_sum" (sum weighted by inverse class frequency)
    """
    from collections import defaultdict

    if aggregation == "class_weighted_sum":
        # Need per-crop (score, file) to read class from .txt
        by_img = defaultdict(list)
        for score, f in scored_files:
            name = os.path.basename(f)
            img_name = name[: name.index("_cropped")]
            txt = f.replace(".npy", ".txt")
            cls_weight = default_class_weight
            if os.path.exists(txt):
                try:
                    with open(txt) as fh:
                        line = fh.readline().strip()
                        if line:
                            cls = int(line.split()[0])
                            cls_weight = (class_weights or {}).get(cls, default_class_weight)
                except Exception:
                    pass
            by_img[img_name].append(score * cls_weight)
        img_scores = [(sum(ds), img) for img, ds in by_img.items()]
    else:
        by_img = defaultdict(list)
        for score, f in scored_files:
            name = os.path.basename(f)
            img_name = name[: name.index("_cropped")]
            by_img[img_name].append(score)

        if aggregation == "max":
            img_scores = [(max(ds), img) for img, ds in by_img.items()]
        elif aggregation == "sum":
            img_scores = [(sum(ds), img) for img, ds in by_img.items()]
        elif aggregation == "mean":
            img_scores = [(np.mean(ds), img) for img, ds in by_img.items()]
        elif aggregation == "crop_weighted":
            img_scores = [(np.mean(ds) * np.sqrt(len(ds)), img) for img, ds in by_img.items()]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    img_scores.sort(key=lambda x: x[0], reverse=True)
    return [img for _, img in img_scores[:k]]


def select_embeddings(
    first_list,
    second_list,
    n=None,
    k=None,
    mode="distance",
    backend="annoy",
    coarse_to_fine=False,
    coarse_k1_mult=4,
    coarse_k2_mult=2,
    coarse_d1_div=8,
    coarse_d2_div=4,
    hnsw_batch_size=1024,
    exact_batch_size=2048,
    granularity_divs=None,
    image_aggregation="max",
    from_fraction=0.0,
    to_fraction=0.0,
    gbm_model_path="gbm_oracle_balanced.pkl",
):
    if mode not in ["distance", "density", "matryoshka_variance", "gbm_oracle"]:
        raise ValueError("`mode` should be 'distance', 'density', 'matryoshka_variance', or 'gbm_oracle'")
    if k is None and n is None:
        raise ValueError("either `n` of `k` should be specified")
    elif k is not None and n is not None:
        raise ValueError("`n` of `k` cannot both be specified")
    if backend not in ["annoy", "hnsw"]:
        raise ValueError("`backend` should be either 'annoy' or 'hnsw'")

    total_embeddings = len(second_list)
    k_int = int(total_embeddings // n) if n is not None else int(k)

    if mode == "gbm_oracle":
        return _select_by_gbm_oracle(
            first_list, second_list, k=k_int,
            from_fraction=from_fraction, to_fraction=to_fraction,
            gbm_model_path=gbm_model_path,
        )

    if mode == "matryoshka_variance":
        return _select_by_granularity_variance(
            first_list, second_list, k=k_int, granularity_divs=granularity_divs
        )

    embedding_dim = get_embedding_dim(first_list)
    # Density uses k-NN (intended: 5-NN); distance uses 1-NN
    nn_k = 5 if mode == "density" else 1
    nn_k = min(nn_k, max(1, len(first_list)))

    k = k_int
    reverse = mode == "distance"

    def _score_files_hnsw(index, files, use_dim: int, nn_k: int, batch_size: int):
        scored = []
        batch_size = int(batch_size) if batch_size is not None else 1024
        batch_size = max(1, batch_size)
        for start in tqdm(range(0, len(files), batch_size)):
            batch_files = files[start : start + batch_size]
            batch = np.empty((len(batch_files), int(use_dim)), dtype=np.float32)
            for i, file in enumerate(batch_files):
                emb = np.load(file).squeeze(0).astype(np.float32)
                batch[i] = _slice_vector(emb, use_dim)
            _, distances = index.knn_query(batch, k=nn_k)
            for i, file in enumerate(batch_files):
                score = _score_from_distances(distances[i].tolist(), mode)
                scored.append((score, file))
        return scored

    if not coarse_to_fine:
        if backend == "annoy":
            print("Building Annoy index from the first list...")
            index = build_annoy_index(first_list, embedding_dim, n_trees=10)
            print("Annoy index built.")
        else:
            print("Building HNSW index from the first list...")
            index = build_hnsw_index(first_list, embedding_dim)
            print("HNSW index built.")

        if mode == "distance":
            print(f"Selecting top {k} embeddings with the largest distances.")
        else:
            print(
                f"Selecting top {k} embeddings by {nn_k}-NN density (inverse-square sum)."
            )

        embeds_with_scores = []
        if backend == "annoy":
            for file in tqdm(second_list):
                emb = np.load(file).squeeze(0).astype(np.float32)
                emb = _slice_vector(emb, embedding_dim)
                _, distances = index.get_nns_by_vector(
                    emb, nn_k, include_distances=True
                )
                score = _score_from_distances(distances, mode)
                embeds_with_scores.append((score, file))
        else:
            embeds_with_scores = _score_files_hnsw(
                index, second_list, embedding_dim, nn_k, hnsw_batch_size
            )
        if image_aggregation == "max":
            sorted_embeds = sorted(embeds_with_scores, key=lambda x: x[0], reverse=reverse)
            res = []
            seen = set()
            for score, f in sorted_embeds:
                name = os.path.basename(f)
                img_name = name[: name.index("_cropped")]
                if img_name in seen:
                    continue
                seen.add(img_name)
                res.append(img_name)
                if len(res) >= k:
                    break
            return res
        else:
            cw, dcw = ({}, 1.0)
            if image_aggregation == "class_weighted_sum":
                print("Computing class weights from labeled pool...")
                cw, dcw = _build_class_weights_from_pool(first_list)
                print(f"  {len(cw)} classes in pool, weight range [{min(cw.values()):.1f}, {max(cw.values()):.1f}]")
            return _aggregate_scores_by_image(embeds_with_scores, k, image_aggregation, cw, dcw)

    # Coarse-to-fine pipeline
    print("Coarse-to-fine selection enabled")
    coarse_k1_mult = float(coarse_k1_mult)
    coarse_k2_mult = float(coarse_k2_mult)
    if coarse_k1_mult < 1 or coarse_k2_mult < 1:
        raise ValueError("coarse_k1_mult and coarse_k2_mult must be >= 1")
    coarse_d1_div = int(coarse_d1_div)
    coarse_d2_div = int(coarse_d2_div)
    if coarse_d1_div < 1 or coarse_d2_div < 1:
        raise ValueError("coarse_d1_div and coarse_d2_div must be >= 1")

    k1 = min(len(second_list), max(1, int(k * coarse_k1_mult)))
    k2 = min(len(second_list), max(1, int(k * coarse_k2_mult)))

    # Stage 1: HNSW on reduced dims
    if not HAS_HNSWLIB:
        raise ImportError("hnswlib not installed; required for coarse-to-fine mode.")
    d1 = max(1, embedding_dim // coarse_d1_div)
    print(f"[Stage 1] Building HNSW index on {d1}/{embedding_dim} dims...")
    index1 = build_hnsw_index(first_list, embedding_dim, use_dim=d1)
    print("[Stage 1] Ranking candidates...")
    scores1 = _score_files_hnsw(index1, second_list, d1, nn_k, hnsw_batch_size)
    stage1_sorted = sorted(scores1, key=lambda x: x[0], reverse=reverse)
    candidates_stage1 = _collect_top_by_image(stage1_sorted, k1)

    # Stage 2: HNSW on reduced dims
    d2 = max(1, embedding_dim // coarse_d2_div)
    print(f"[Stage 2] Building HNSW index on {d2}/{embedding_dim} dims...")
    index2 = build_hnsw_index(first_list, embedding_dim, use_dim=d2)
    print("[Stage 2] Re-ranking candidates...")
    scores2 = _score_files_hnsw(index2, candidates_stage1, d2, nn_k, hnsw_batch_size)
    stage2_sorted = sorted(scores2, key=lambda x: x[0], reverse=reverse)
    candidates_stage2 = _collect_top_by_image(stage2_sorted, k2)

    # Stage 3: Exact cosine KNN on full dims
    print(
        f"[Stage 3] Exact re-ranking on full vectors (dim = {embedding_dim}) (cosine distance)..."
    )
    from sklearn.neighbors import NearestNeighbors

    first_matrix = _load_matrix(first_list, use_dim=None, normalize=True)
    nn_exact = NearestNeighbors(
        n_neighbors=int(nn_k), metric="cosine", algorithm="brute", n_jobs=-1
    )
    nn_exact.fit(first_matrix)

    scores3 = []
    exact_batch_size = int(exact_batch_size) if exact_batch_size is not None else 2048
    exact_batch_size = max(1, exact_batch_size)
    for start in tqdm(range(0, len(candidates_stage2), exact_batch_size)):
        batch_files = candidates_stage2[start : start + exact_batch_size]
        batch = _load_matrix(batch_files, use_dim=None, normalize=True)
        distances, _ = nn_exact.kneighbors(
            batch, n_neighbors=int(nn_k), return_distance=True
        )
        for i, file in enumerate(batch_files):
            score = _score_from_distances(distances[i].tolist(), mode)
            scores3.append((score, file))
    if image_aggregation == "max":
        stage3_sorted = sorted(scores3, key=lambda x: x[0], reverse=reverse)
        res = []
        seen = set()
        for score, f in stage3_sorted:
            name = os.path.basename(f)
            img_name = name[: name.index("_cropped")]
            if img_name in seen:
                continue
            seen.add(img_name)
            res.append(img_name)
            if len(res) >= k:
                break
        return res
    else:
        cw, dcw = ({}, 1.0)
        if image_aggregation == "class_weighted_sum":
            print("Computing class weights from labeled pool...")
            cw, dcw = _build_class_weights_from_pool(first_list)
            print(f"  {len(cw)} classes in pool, weight range [{min(cw.values()):.1f}, {max(cw.values()):.1f}]")
        return _aggregate_scores_by_image(scores3, k, image_aggregation, cw, dcw)


def select_embeddings_voting(
    first_lists,
    second_lists,
    k,
    mode="distance",
    backend="annoy",
    coarse_to_fine=False,
    coarse_k1_mult=4,
    coarse_k2_mult=2,
    coarse_d1_div=8,
    coarse_d2_div=4,
    hnsw_batch_size=1024,
    exact_batch_size=2048,
    image_aggregation="max",
):
    """
    Run selection separately for multiple embedding spaces (e.g. 3 feature maps),
    then combine selected image lists by voting.

    Vote for an image is how many per-space selections contain it (1..N).
    Ties are broken by average rank (lower is better).
    """
    if not isinstance(first_lists, (list, tuple)) or not isinstance(
        second_lists, (list, tuple)
    ):
        raise TypeError("first_lists and second_lists must be lists/tuples of lists")
    if len(first_lists) != len(second_lists):
        raise ValueError("first_lists and second_lists must have the same length")
    if len(first_lists) == 0:
        return []

    per_space_selected = []
    for fl, sl in zip(first_lists, second_lists):
        per_space_selected.append(
            select_embeddings(
                fl,
                sl,
                k=k,
                mode=mode,
                backend=backend,
                coarse_to_fine=coarse_to_fine,
                coarse_k1_mult=coarse_k1_mult,
                coarse_k2_mult=coarse_k2_mult,
                coarse_d1_div=coarse_d1_div,
                coarse_d2_div=coarse_d2_div,
                hnsw_batch_size=hnsw_batch_size,
                exact_batch_size=exact_batch_size,
            )
        )

    # votes[img] = count of lists containing img
    # ranks[img] = list of ranks (0-based) across spaces where present
    votes = {}
    ranks = {}
    for sel in per_space_selected:
        for r, img in enumerate(sel):
            votes[img] = votes.get(img, 0) + 1
            ranks.setdefault(img, []).append(r)

    # Sort by:
    # 1) vote desc (3->2->1)
    # 2) average rank asc (better rank first)
    # 3) name for determinism
    def avg_rank(img):
        rs = ranks.get(img, [])
        return float(sum(rs) / max(1, len(rs)))

    ordered = sorted(votes.keys(), key=lambda img: (-votes[img], avg_rank(img), img))
    return ordered[: int(k)]
