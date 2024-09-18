import numpy as np
from pathlib import Path
from typing import Union, List, Tuple
from utils import TextStyles, stylish_text
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import IncrementalPCA
from torch.utils.data.dataloader import default_collate
import shutil
import joblib
import umap
import argparse



class VectorDataset(Dataset):

    def __init__(self, data_source: Union[str, Path, List[Union[str, Path]]]):
        self.data_source = data_source
        self.file_names = self._load_stored_embeddings()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        file_path = self.file_names[index]
        vector = np.load(file_path).squeeze()  # as files are .npy stored vectors
        return vector, index

    def _load_stored_embeddings(
        self,
    ) -> List[Path]:
        if type(self.data_source) in (str, Path):
            embedding_files = [
                file for file in Path(self.data_source).glob("*.npy") if file.is_file()
            ]
        elif type(self.data_source) == list:
            assert all(type(item) in (str, Path) for item in self.data_source)
            embedding_files = []
            for item in self.data_source:
                embedding_files += [
                    file for file in Path(item).glob("*.npy") if file.is_file()
                ]
        else:
            raise ValueError(
                "unexpected data source format. Should be either path to dir with embeddings or a list of such dirs"
            )
        return embedding_files


def custom_collate_fn(batch):
    # Split data and indices
    data, indices = zip(*batch)
    # Use default_collate to efficiently collate the data part
    batch_data = default_collate(data)
    # No need to collate indices as they are already a tuple of scalars
    return batch_data, torch.tensor(indices)


class EmbeddingPoolPreprocessor:
    """
    reduce dimensions of initial embeddings
    """

    def __init__(
        self,
        embeddings_source: Union[str, Path, List[Union[str, Path]]],
        output_dir: Union[str, Path],
        target_length: int = 512,
        batch_size: int = 512,
        copy_image=True,
    ):
        self.copy_image = copy_image
        self.embeddings_source = embeddings_source
        print(stylish_text("Collecting embedding file paths", TextStyles.OKBLUE))
        embeddings_dataset = VectorDataset(embeddings_source)
        self.embeddings_dataset = embeddings_dataset
        self.init_embedding_length = embeddings_dataset[0][0].shape[
            0
        ]  # actually can use size as it is 1-dim vector
        self.target_length = target_length
        self.dataloader = DataLoader(
            embeddings_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        self.output_dir = Path(output_dir)
        self.expected_pca_model_path = Path(f"{output_dir}_PCA_model.joblib")

    def run_dimension_reduction(self, mode: str = "PCA") -> None:
        """
        @param mode: str (anycase) in ("PCA", "UMAP")
        """
        if mode.lower() == "pca":
            return self.__run_pca_dimension_reduction()
        elif mode.lower() == "umap":
            return self.__run_umap_dimension_reduction()
        else:
            raise NotImplementedError(
                f"given dimension reduction method: ({mode}) is not implemented"
            )

    def __run_pca_dimension_reduction(self) -> None:
        print(
            stylish_text("Running PCA for dimensionality reduction", TextStyles.OKBLUE)
        )

        # Initialize PCA model
        if self.expected_pca_model_path.is_file():
            pca = joblib.load(self.expected_pca_model_path)
        else:
            pca = IncrementalPCA(n_components=self.target_length)

            # Fit PCA on the entire dataset using a loop to avoid memory issues
            print(stylish_text("Fitting PCA model", TextStyles.OKBLUE))
            for batch, _ in tqdm(self.dataloader, desc="Fitting PCA"):
                if (
                    batch.shape[0] < self.target_length
                ):  # Skip batches smaller than n_components
                    print(
                        stylish_text(
                            f"Batch of different size={batch.shape[0]} encountered during PCA fit phase. Skipping. Not a problem if #batches is not low",
                            TextStyles.WARNING,
                        )
                    )
                    continue
                batch = (batch - torch.mean(batch, 0)) / (
                    torch.sqrt(torch.var(batch, 0)) + 1e-5
                )
                # print(f"mean: {torch.mean(batch, 0)}")
                # print(f"var: {torch.var(batch, 0)}")
                pca.partial_fit(batch)  # Use partial_fit to accommodate large datasets
            joblib.dump(pca, self.expected_pca_model_path)

        # Transform embeddings to reduced dimensionality and save them
        self.output_dir.mkdir(exist_ok=True, parents=True)

        print(stylish_text("Transforming embeddings and saving", TextStyles.OKBLUE))
        for batch, indices in tqdm(self.dataloader, desc="Transforming embeddings"):
            reduced_embeddings = pca.transform(batch)

            for idx, reduced_embedding in zip(indices, reduced_embeddings):
                # Use idx to access the global index in the dataset
                orig_file_path = self.dataloader.dataset.file_names[idx]
                orig_file_name = Path(orig_file_path).name
                save_path = self.output_dir / orig_file_name
                np.save(
                    save_path, reduced_embedding[np.newaxis, ...]
                )  # to be in consistency with large embedding storing format

    def synchronize_input_output_file_structure(
        self,
        embeddings_source: Union[str, Path, List[Union[str, Path]]],
        output_dir: Union[str, Path],
    ) -> None:
        """
        adding image crops to embeddings and annotation files as in original folder
        if embeddings_source was a list of dir paths -> let's split embeddings accordingly in output_dir
        """
        print(stylish_text("synchronize I/O file structure", TextStyles.OKBLUE))
        files_in_output_dir = VectorDataset(output_dir).file_names

        if type(embeddings_source) == list:
            file_name_to_file_path = {
                file.name: file
                for source in embeddings_source
                for file in VectorDataset(source).file_names
            }

            for file in files_in_output_dir:
                corresponding_parent_folder = file_name_to_file_path[
                    file.name
                ].parent.name
                dst_path = Path(output_dir) / corresponding_parent_folder / file.name
                dst_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.move(str(file), str(dst_path))

            files_in_output_dir = [
                file
                for file in Path(output_dir).rglob("*.*")
                if (file.is_file() and (file.suffix == ".npy"))
            ]

        elif type(embeddings_source) in (str, Path):
            file_name_to_file_path = {
                file.name: file for file in VectorDataset(embeddings_source).file_names
            }

        else:
            raise ValueError(
                stylish_text(
                    "unknown emnbedding_source format: either str, Path to dir or List of such paths"
                ),
                TextStyles.FAIL,
            )

        print(stylish_text("copying annotations and image crops", TextStyles.OKBLUE))
        for file in tqdm(files_in_output_dir):
            corresponding_source_file = file_name_to_file_path[file.name]
            if self.copy_image:
                for suf in (".jpeg", ".jpg", "png"):
                    corresponding_image = corresponding_source_file.with_suffix(suf)
                    if corresponding_image.is_file():
                        break
                shutil.copy(
                    str(corresponding_image),
                    str(file.parent / corresponding_image.name),
                )
            corresponding_annotations = corresponding_source_file.with_suffix(".txt")
            shutil.copy(
                str(corresponding_annotations),
                str(file.parent / corresponding_annotations.name),
            )

    def __run_umap_dimension_reduction(self) -> None:
        print(
            stylish_text("Running UMAP for dimensionality reduction", TextStyles.OKBLUE)
        )

        print(stylish_text("loading vectors into UMAP", TextStyles.OKBLUE))
        all_vectors = []
        for item in tqdm(self.embeddings_dataset):
            embedding_vector, idx = item
            # embedding_filename = self.embeddings_dataset.file_names[idx]
            all_vectors.append(embedding_vector)
        all_vectors = np.array(all_vectors)

        print(stylish_text("Fitting UMAP model", TextStyles.OKBLUE))
        # Fit UMAP and transform the data
        reducer = umap.UMAP(n_components=512, random_state=42)
        reduced_data = reducer.fit_transform(all_vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    embeddings_source = [
        # "/media/smarkov1001/storage_ssd/embedding_analysis_data/runs/4_layers/run64_on_train",
        # "/media/smarkov1001/storage_ssd/embedding_analysis_data/runs/4_layers/run64_on_val"
        # "/media/smarkov1001/storage_ssd/embedding_analysis_data/runs/3_layers_22concats_0_1_2/train",
        # "/media/smarkov1001/storage_ssd/embedding_analysis_data/runs/3_layers_22concats_0_1_2/val",
        "/home/setupishe/bel_conf/remainder_embeds_0.2"
    ]
    output_dir = "/home/setupishe/bel_conf/remainder_embeds_reduced_0.2"
    epp = EmbeddingPoolPreprocessor(
        embeddings_source,
        output_dir,
        target_length=512,
        batch_size=512,
    )
    epp.run_dimension_reduction(mode="PCA")
    epp.synchronize_input_output_file_structure(embeddings_source, output_dir)
