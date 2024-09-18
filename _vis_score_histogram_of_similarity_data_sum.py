import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List, Union
import yaml
from tqdm import tqdm
import math


ALLOWED_IMAGE_SUFFIXES = ".jpeg", ".jpg", ".png"

def plot_score_distribution(scores: Dict[str, float], num_bins: int = 20) -> None:
    # Extract scores from the dictionary
    score_values = np.array(list(scores.values()))
    
    # Plot histogram with the specified number of bins
    plt.figure(figsize=(10, 6))
    plt.hist(score_values, bins=num_bins, alpha=0.7, color='blue', edgecolor='black')
    
    # Adding titles and labels
    plt.title('Distribution of Scores')
    plt.xlabel('Score Values')
    plt.ylabel('Number of Files')
    
    # Show grid and the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def count_embeddings(scores: Dict[str, float], threshold: float, mode: str='right') -> Tuple[int, Dict[str, float]]:
    method = "__le__" if mode == 'left' else "__ge__"
    res = {}
    for p, s in tqdm(scores.items()):
        if getattr(s, method)(threshold):
            res[p] = s
    return len(res), res


def count_original_images(scores: Dict[str, float], folder_with_original_images: Union[str, Path]) -> Tuple[int, Dict[str, float]]:
    res = {}
    for path_to_e, score in tqdm(scores.items()):
        original_stem = str(Path(path_to_e).name).split("_cropped")[0]
        for suf in ALLOWED_IMAGE_SUFFIXES:
            original_image = Path(folder_with_original_images) / f"{original_stem}{suf}"
            if original_image.is_file():
                break
        if (original_image is None) or (not original_image.is_file()):
            raise FileNotFoundError(f"original image not found for: {path_to_e}")

        res_key = original_image.name
        try:
            current_score = res[res_key]
        except KeyError:
            current_score = -math.inf
        
        if score >= current_score:
            res[res_key] = score
    return len(res), res





if __name__ == "__main__":
    train_dir = "/home/smarkov1001/sm_3.46/outputs/human_head_detection_yolov8_dataset/train"
    dict_src = "/home/smarkov1001/NVI/scripts/TMP/similarity_pairs/similarity_data_sum.yaml"
    scores = yaml.safe_load(Path(dict_src).read_text())
    # plot_score_distribution(scores, num_bins=100)
    _, embeddings_above_theshold = count_embeddings(scores, 300)
    n, original_images = count_original_images(embeddings_above_theshold, train_dir)
    print(n)