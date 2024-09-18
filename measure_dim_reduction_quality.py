"""
the idea is to try to see how different angles between same vectors if they are in full-dimension form and if they are in reduced dim form
"""
from typing import Union
from pathlib import Path
from utils import load_stored_embeddings
from tqdm import tqdm
import numpy as np


def measure_dim_reduction_quality(
        vector_src_dir: Union[str, Path],
        vector_mod_dir: Union[str, Path],
        class_of_interest: int
):
    _, embeddings_src, _ = load_stored_embeddings(vector_src_dir, class_of_interest)
    _, embeddings_mod, _ = load_stored_embeddings(vector_mod_dir, class_of_interest)
    embeddings_src = sorted(embeddings_src)
    embeddings_mod = sorted(embeddings_mod)
    print("Checking that every vector from src dir have corresponding vector in mod dir")
    if not len(embeddings_src) == len(embeddings_mod):
        raise RuntimeError("list of short and long vectors have different length!")
    for v1, v2 in tqdm(zip(embeddings_src, embeddings_mod), total=len(embeddings_src)):
        if v1.name != v2.name:
            raise RuntimeError(f"{v1.name} vector from embeddings_src dir have no corresponding vector in embeddings_mod dir")
    print("Measuring angles between each two vectors in both sets. Comparing results.")
    for i in tqdm(range(len(embeddings_src))):
        for j in range(i + 1, len(embeddings_src)):
            src_v1 = np.load(embeddings_src[i])
            src_v2 = np.load(embeddings_src[j])
            mod_

if __name__ == "__main__":
    long_vectors_dir = "/media/smarkov1001/storage_ssd/embedding_analysis_data/runs/3_layers_22concats_0_1_2/train"
    short_vectors_dir = "/media/smarkov1001/storage_ssd/embedding_analysis_data/runs/3_layers_22concats_0_1_2/REDUCED512_UMAP/train"
    measure_dim_reduction_quality(
        vector_src_dir=long_vectors_dir,
        vector_mod_dir=short_vectors_dir,
        class_of_interest=1
    )