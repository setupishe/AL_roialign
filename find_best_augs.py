"""
1. Scan embeddings computed from train and val folders
2. Determine which embeddings for val are actually FP
3. For each FP from val: find N closest embeddings in train directory
4. Take original images where they came from, annotations that are defining them
5. Apply albumentations to the image, get modified annotations
6. Get embeddings from these annotations
7. Goal is to shift center mass of N closest embeddings to FP point
repeat search on a grid
"""
from typing import Union, Sequence, Optional, Dict, List, Tuple
from pathlib import Path, PosixPath
from cv2 import NORM_INF
from produce_detection_embeddings import YoloEmbeddingsProducer
import joblib
from utils import load_stored_embeddings, load_ground_truth_markup_for_image_crop_file_paths, assign_detection_status_to_prediction, any_suf_image_path
from utils import stylish_text, TextStyles, Annotation, Bbox, file_to_annotation
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch
import torch.nn.functional as F
import albumentations as A
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import yaml


class EmbeddingDataset(Dataset):

    def __init__(self, embedding_path_list: List[Path]):
        self.embedding_path_list = embedding_path_list
    
    def __len__(self):
        return len(self.embedding_path_list)
    
    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        res: np.ndarray = np.load(self.embedding_path_list[index])
        res = res.squeeze(axis=0)
        return res, index
    

def custom_collate_fn(batch):
    # Split data and indices
    data, indices = zip(*batch)
    # Use default_collate to efficiently collate the data part
    batch_data = default_collate(data)
    # No need to collate indices as they are already a tuple of scalars
    return batch_data, torch.tensor(indices)  


class AugmentationOptimizer:

    def __init__(
        self,
        embeddings_val_dir: str,
        embeddings_train_dir: str,
        ground_truth_val_dir: str, # what embeddings are FP?
        ground_truth_train_dir: str, # to take original images and apply albumentations
        pca_model_path: str,
        class_of_interest: int,
        onnx_model_path: Union[str, Path],
        netron_layer_names: Sequence[str],
        providers: List[str]=['CUDAExecutionProvider']
    ):
        self.e_val_dir = Path(embeddings_val_dir)
        self.e_train_dir = Path(embeddings_train_dir)
        self.gt_val_dir = Path(ground_truth_val_dir)
        self.gt_train_dir = Path(ground_truth_train_dir)
        self.pca_model = joblib.load(pca_model_path)
        self.class_of_interest = class_of_interest

        self.embedding_producer = YoloEmbeddingsProducer(onnx_model_path, netron_layer_names, None, providers)

    def assign_fp_coeff_to_each_train_embedding(
            self,
            output_yaml: Union[str, Path], # returns dict and also dumps it to output_yaml path
            mode: Optional[str]=None # None: for each train embedding have a list of len N_fp of similarity values (i.e. all values)
        ) -> Dict[str, float]:
        """
        @param mode: Optional[str]:
                        1. None: for each train embedding have a list of len N_fp of similarity values (i.e. all values)
                        2. 'sum' for each train embedding have a single value that is a sum of all similarity values across all FP embeddings
                        3. TODO implement whatever you want
        @returns: dict {embedding file path : cumulative similarity value} if mode is not None. If mode is None returns None
                                              and just writes to file as dict will be to large to handle
        """
        ## ==== which embeddings from val are actually pure FP?
        print(stylish_text("loading validation embeddings and annotations", TextStyles.OKBLUE))
        val_annotations, val_embedding_files, val_image_paths = load_stored_embeddings(
            self.e_val_dir, 
            self.class_of_interest
        )

        print(stylish_text("loading ground truth annotations", TextStyles.OKBLUE))
        gt_files = load_ground_truth_markup_for_image_crop_file_paths(val_image_paths, self.gt_val_dir)

        print(stylish_text("searching for FP validation embeddings", TextStyles.OKBLUE))
        only_fp_detections: List[Tuple[List[Path], List[Path], List[Path]]] = []
        for val_annotation, val_embedding_file, val_image_path, gt_file in tqdm(
            zip(val_annotations, val_embedding_files, val_image_paths, gt_files), 
            total=len(val_image_paths)
            ):
            prediction_status: str = assign_detection_status_to_prediction(val_annotation, gt_file, tp_treshhold=0.05)
            if prediction_status == 'fp':
                only_fp_detections.append((val_annotation, val_embedding_file, val_image_path))
        
        only_fp_detections = sorted(only_fp_detections, key=lambda sublist: sublist[2])

        val_annotations, val_embedding_files, val_image_paths = zip(*only_fp_detections)

        print(stylish_text("loading embedding vectors for val FP detections", TextStyles.OKBLUE))
        val_embedding_matrix = torch.tensor(
            np.array(
                [np.load(val_embedding_file) for val_embedding_file in val_embedding_files]
            ).squeeze(axis=1)
        ,dtype=torch.double)
        val_embedding_matrix_norm = F.normalize(val_embedding_matrix, p=2, dim=1)
        
        ## ==== which N=train_cluster_size embeddings from train are closest to val fp embeddings?
        print(stylish_text("loading train annotations", TextStyles.OKBLUE))
        train_annotations, train_embedding_files, train_image_paths = load_stored_embeddings(
            self.e_train_dir, 
            self.class_of_interest
        )

        train_dataset = EmbeddingDataset(train_embedding_files)
        train_dataloader = DataLoader(train_dataset, batch_size=len(only_fp_detections), shuffle=False, collate_fn=custom_collate_fn)

        restore_path = lambda index_in_dataset: train_dataloader.dataset.embedding_path_list[index_in_dataset]
        global_similarity_dict = {}
        if mode is None:
            with open(output_yaml, 'w') as stream:
                pass
        print(stylish_text("computing cosine similarity index for vector batches", TextStyles.OKBLUE))
        for batch, indices in tqdm(train_dataloader):
            if batch.shape[0] != len(val_embedding_files):
                continue
            # here batch is tensor of shape N_fp, 512
            batch_norm = F.normalize(batch, p=2, dim=1)
            similarity_matrix = torch.mm(batch_norm, val_embedding_matrix_norm.T) # {N_fp(batch_size) x 512} X {512, N_fp(#of FP)}
            ## === so similarity matrix is: for each vector from train batch we have N_fp values of similarity relative to all FPs
            filenames = [str(restore_path(i)) for i in indices]
            if mode == 'sum':
                similarity_matrix = similarity_matrix.sum(dim=1)
            batch_dict = dict(zip(filenames, similarity_matrix.tolist()))
            if mode is not None:
                global_similarity_dict.update(batch_dict)
            else:
                with open(output_yaml, 'a') as stream:
                    for filename, similarity_values_list in batch_dict.items():
                        yaml_block = f"{filename}:\n" + "\n".join([f"- {val}" for val in similarity_values_list]) + "\n"
                        stream.write(yaml_block)
        
        if mode is not None:
            with open(output_yaml, 'w') as stream:
                yaml.safe_dump(global_similarity_dict, stream, indent=2, width=1000)
            
        return global_similarity_dict if mode is not None else None
    

    @DeprecationWarning
    def make_distribution_graphs_of_train_similarities_against_each_fp_embedding(
            self,
            similarity_data_yaml: Union[str, Path],
            graph_output_dir: Union[str, Path]
        ) -> Dict[str, float]:
        number_of_fps = None
        with open(similarity_data_yaml, 'r') as stream:
            counter = 0
            for line in stream:
                if counter != 0:
                    if not line.startswith("-"):
                        break
                if line.startswith("-"):
                    counter += 1
            number_of_fps = counter
        
        for fp_idx in tqdm(range(number_of_fps)):
            graph_data = []
            with open(similarity_data_yaml, 'r') as stream:
                for line in stream:
                    if not line.startswith("-"):
                        block_counter = -1
                    else:
                        block_counter += 1
                    if block_counter == fp_idx:
                        value = line.replace("- ", "").replace("\n", "").strip()
                        graph_data.append(value)
            for item in graph_data:
                print(item)
            print(f"{len(graph_data)=}")
            break


    def find_n_nearest_train_images_to_each_val_fp(
            self, 
            train_cluster_size: int=50,
            save: bool=False
            ):
        """
        @returns: List[Path], np.ndarray
                  First list is a list of image crops that are FP detections on validation set
                  Array is of len = train_cluster_size, 
                  where each row have a len = #(fp in val_annotations detections)
                  so basically in each column we have train_cluster_size of image_crops from train 
                  that are closest to correspondning FP from first returned list
        """
        ## ==== which embeddings from val are actually pure FP?
        print(stylish_text("loading validation embeddings and annotations", TextStyles.OKBLUE))
        val_annotations, val_embedding_files, val_image_paths = load_stored_embeddings(
            self.e_val_dir, 
            self.class_of_interest
        )

        print(stylish_text("loading ground truth annotations", TextStyles.OKBLUE))
        gt_files = load_ground_truth_markup_for_image_crop_file_paths(val_image_paths, self.gt_val_dir)

        print(stylish_text("searching for FP validation embeddings", TextStyles.OKBLUE))
        only_fp_detections: List[Tuple[List[Path], List[Path], List[Path]]] = []
        for val_annotation, val_embedding_file, val_image_path, gt_file in tqdm(
            zip(val_annotations, val_embedding_files, val_image_paths, gt_files), 
            total=len(val_image_paths)
            ):
            prediction_status: str = assign_detection_status_to_prediction(val_annotation, gt_file, tp_treshhold=0.05)
            if prediction_status == 'fp':
                only_fp_detections.append((val_annotation, val_embedding_file, val_image_path))
        
        only_fp_detections = sorted(only_fp_detections, key=lambda sublist: sublist[2])

        val_annotations, val_embedding_files, val_image_paths = zip(*only_fp_detections)

        print(stylish_text("loading embedding vectors for val FP detections", TextStyles.OKBLUE))
        val_embedding_matrix = torch.tensor(
            np.array(
                [np.load(val_embedding_file) for val_embedding_file in val_embedding_files]
            ).squeeze(axis=1)
        ,dtype=torch.double)
        val_embedding_matrix_norm = F.normalize(val_embedding_matrix, p=2, dim=1)
        
        ## ==== which N=train_cluster_size embeddings from train are closest to val fp embeddings?
        print(stylish_text("loading train annotations", TextStyles.OKBLUE))
        train_annotations, train_embedding_files, train_image_paths = load_stored_embeddings(
            self.e_train_dir, 
            self.class_of_interest
        )
        train_dataset = EmbeddingDataset(train_embedding_files)
        print(f"{len(train_dataset)=}")
        train_dataloader = DataLoader(train_dataset, batch_size=len(only_fp_detections), shuffle=False, collate_fn=custom_collate_fn)

        print(stylish_text("computing cosine similarity index for vector batches", TextStyles.OKBLUE))
        max_stored_closest_n = torch.zeros(size=(train_cluster_size, len(val_embedding_files)), dtype=torch.double)
        max_stored_closest_n -= 1
        indices_of_max_stored_closest_n = torch.zeros_like(max_stored_closest_n, dtype=torch.long)
        for batch, indices in tqdm(train_dataloader):
            if batch.shape[0] != len(val_embedding_files):
                continue
            # here batch is tensor of shape N_fp, 512
            batch_norm = F.normalize(batch, p=2, dim=1)
            similarity_matrix = torch.mm(batch_norm, val_embedding_matrix_norm.T)
            closest_n, closest_n_indices = torch.topk(similarity_matrix, train_cluster_size, dim=0, largest=True)
            max_similarity_embeds_indices_in_dataset = torch.gather(
                indices.expand(closest_n_indices.shape[0], -1),
                dim=1,
                index=closest_n_indices
            )
            max_update_mask = closest_n > max_stored_closest_n
            max_stored_closest_n[max_update_mask] = closest_n[max_update_mask]
            indices_of_max_stored_closest_n[max_update_mask] = max_similarity_embeds_indices_in_dataset[max_update_mask]
        

        restore_path = lambda index_in_dataset: train_dataloader.dataset.embedding_path_list[index_in_dataset]

        matrix_of_file_paths = []  # matrix of shape train_cluster_size x len(val_embedding_files)
        for row in indices_of_max_stored_closest_n:
            paths_in_a_row = [restore_path(idx) for idx in row.tolist()]
            matrix_of_file_paths.append(paths_in_a_row)
        
        matrix_of_file_paths = np.array(matrix_of_file_paths, dtype=object)

        if save:
            save_dir = Path(__file__).resolve().parent
            np.save(save_dir / "similarity_filepath_matrix.npy", matrix_of_file_paths)
            with open(save_dir / "similarity_corresponding_fp_crops.txt", 'w') as stream:
                for path in val_embedding_files:
                    stream.write(str(path) + "\n")

        return val_embedding_files, matrix_of_file_paths, max_stored_closest_n.cpu().numpy()
    
    def visualize_find_n_nearest_train_images_to_each_val_fp(
        self, 
        output_dir: Union[str, Path],
        train_cluster_size: int=50      
    ):
        import shutil
        val_fp_embedding_file_paths, matrix_of_file_paths_to_nearest_train_embeddings, matrix_of_similarity_data = self.find_n_nearest_train_images_to_each_val_fp(
            train_cluster_size, 
            save=False
        )
        for val_idx, val_fp_file in tqdm(enumerate(val_fp_embedding_file_paths)):
            for suf in (".jpeg", ".jpg", ".png"):
                val_fp_image = val_fp_file.with_suffix(suf)
                if val_fp_image.is_file():
                    break
            fp_dir = Path(output_dir) / "fp" / str(val_idx)
            fp_dir.mkdir(exist_ok=True, parents=True)
            similarity_dir = Path(output_dir) / "similar_train" / str(val_idx)
            similarity_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(val_fp_image, fp_dir / val_fp_image.name)
            for train_emb, similarity_coeff in zip(
                list(matrix_of_file_paths_to_nearest_train_embeddings[:, val_idx]),
                list(matrix_of_similarity_data[:, val_idx])
            ):
                for suf in (".jpeg", ".jpg", ".png"):
                    train_image = train_emb.with_suffix(suf)
                    if train_image.is_file():
                        break     
                shutil.copy(train_image, similarity_dir/ f"{round(similarity_coeff, 5)}T_T{train_image.name}")           
                
    
    def find_best_augmentations_for_nearest_inputs_to_given_vals(
            self,
            train_cluster_size: int=50,
            save_path: Optional[Union[str, Path]]=None,
            mode='bayes' # in 'bayes', 'grid'
    ): 

        val_fp_embedding_file_paths, matrix_of_file_paths_to_nearest_train_embeddings = self.find_n_nearest_train_images_to_each_val_fp(
            train_cluster_size, save=False
        )

        print(stylish_text("Searching for best augmentations for validation false positives", TextStyles.OKBLUE))
        val_path_to_best_augmentations = {}
        for val_sample_idx, val_embedding_path in enumerate(tqdm(val_fp_embedding_file_paths)): # TODO: will remove slice after tuning
            train_embedding_paths = matrix_of_file_paths_to_nearest_train_embeddings[:, val_sample_idx]
            self.__find_best_augmentations_for_nearest_inputs_to_given_vals(
                val_embedding_path,
                train_embedding_paths,
                val_path_to_best_augmentations,
                mode
            )

            # dumping dict every iteration as algorithm is crashing
            if (save_path is not None) and (type(save_path) in (str, PosixPath)):
                save_path = Path(save_path)
                save_path.parent.mkdir(exist_ok=True, parents=True)
                with open(save_path, 'w') as stream:
                    yaml.safe_dump(val_path_to_best_augmentations, stream, indent=2, width=1000)
        return val_path_to_best_augmentations

    def __find_best_augmentations_for_nearest_inputs_to_given_vals(
            self, 
            val_embedding_path: Path,
            train_embedding_paths: np.ndarray,
            val_path_to_best_augmentations: dict,
            mode: str
        ):
        """
        extra inner function to ensure memory freeing
        """
        val_embedding = np.load(val_embedding_path)
        
        train_annotations = [file_to_annotation(file.with_suffix(".txt"))[0] for file in train_embedding_paths]
        
        gt_train_image_paths = [
            any_suf_image_path(self.gt_train_dir / f"{file.stem.split('_cropped')[0]}") 
            for file in train_embedding_paths
        ]
        assert None not in gt_train_image_paths

        best_parameters_finding_func = self.search_optimal_parameters_on_grid if mode == 'grid' else self.search_optimal_parameters_on_space
        best_parameters = best_parameters_finding_func(
            gt_train_image_paths,
            annotations=[annotation for annotation in train_annotations],
            target_embedding=val_embedding
        )
        val_path_to_best_augmentations[str(val_embedding_path)] = {"train_images" : [], "best_augmentation_params" : best_parameters}
        val_path_to_best_augmentations[str(val_embedding_path)]["train_images"].extend(list(map(str, gt_train_image_paths)))

    def _test_copy_fp_and_corresponding_train_crops_to_separate_dir(self, folder):
        """
        this method is a stub as I need result super fast
        """
        train_similarities = np.load("/home/smarkov1001/NVI/scripts/humans/similarity_filepath_matrix.npy", allow_pickle=True)
        fp = Path("/home/smarkov1001/NVI/scripts/humans/similarity_corresponding_fp_crops.txt").read_text().split("\n")[0]
        fp = Path(fp)
        
        train_embeddings = train_similarities[:, 0]

        fp_folder = Path(folder) / "fp"
        train_folder = Path(folder) / "train"

        fp_folder.mkdir(exist_ok=True, parents=True)
        train_folder.mkdir(exist_ok=True, parents=True)

        import shutil

        shutil.copy(fp, fp_folder / fp.name)
        shutil.copy(Path(fp).with_suffix(".txt"), fp_folder / (Path(fp).with_suffix(".txt")).name)
        for tag in (".jpeg", ".jpg", ".png"):
                src_path = Path(fp).with_suffix(tag)
                if not src_path.is_file():
                    continue
                shutil.copy(src_path, fp_folder / src_path.name)
        
        for train_path in train_embeddings:
            shutil.copy(train_path, train_folder / train_path.name)
            shutil.copy(Path(train_path).with_suffix(".txt"), train_folder / (Path(train_path).with_suffix(".txt")).name)   
            for tag in (".jpeg", ".jpg", ".png"):
                    src_path = Path(train_path).with_suffix(tag)
                    if not src_path.is_file():
                        continue
                    shutil.copy(src_path, train_folder / src_path.name)


    # def _find_augs(self):
    #     """
    #     STUB
    #     """
    #     # NOTE that train annotations is a list of single annotations, i.e. each train_image_path have one corresponding annotation
    #     train_annotations, train_embedding_files, train_image_paths = load_stored_embeddings( 
    #         "/home/smarkov1001/NVI/scripts/humans/testing_case/train", 
    #         self.class_of_interest
    #     ) ## here are only humans here, class_index = 1

    #     val_annotations, val_embedding_files, val_image_paths = load_stored_embeddings(
    #         "/home/smarkov1001/NVI/scripts/humans/testing_case/fp", 
    #         self.class_of_interest
    #     )  

    #     val_embedding_vector = np.load(val_embedding_files[0])  # 1 x 512 vector

    #     ## collecting ground truth train files
    #     sufs = ".jpeg", ".jpg", ".png"
    #     ground_truth_train_files = []
    #     for file in train_image_paths:
    #         path_to_original_image = str(Path(self.gt_train_dir) / file.name.split("_cropped")[0])
    #         for suf in sufs:
    #             path_to_original_image_with_suf = Path(path_to_original_image + suf)
    #             if not path_to_original_image_with_suf.is_file():
    #                 continue
    #             ground_truth_train_files.append(path_to_original_image_with_suf)

    #     best_parameters = self.search_optimal_parameters(
    #         ground_truth_train_files,
    #         annotations=[annotation for annotation in train_annotations],
    #         target_embedding=val_embedding_vector
    #     )


    def search_optimal_parameters_on_grid(self, image_paths: List[Path], annotations: List[Annotation], target_embedding: np.ndarray):

        parameter_grid = self.__generate_parameter_grid()

        best_similarity = -1
        best_parameters = None

        images = [Image.open(image_path) for image_path in image_paths]

        for params in tqdm(parameter_grid, total=parameter_grid.shape[0]):
            similarity = -1
            for idx, annotation in enumerate(annotations):
                image = images[idx]
                image_array = np.array(image)
                transformation_result = self.__apply_transformation(image_array, annotation, params)
                if transformation_result is not None:
                    transformed_image, transformed_bbox, class_index = transformation_result
                    x1, y1, x2, y2 = transformed_bbox
                    xc = (x1 + x2) / 2
                    yc = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    transformed_annotation = Annotation(
                        class_index,
                        Bbox(xc, yc, w, h).normalize(image.size),
                        confidence=annotation.confidence
                    )
                    embedding = self.produce_compressed_detection_embedding(transformed_image, transformed_annotation)
                    similarity += cosine_similarity(embedding, target_embedding)[0][0]  # function produces matrix which is 1x1 matrix in my case
                else:
                    similarity += -1
            similarity /= len(annotations)
            if similarity > best_similarity:
                best_similarity = similarity
                # TODO:
                best_parameters = {'rotation': float(params[0]), 'translate': float(params[1]), 'scale': float(params[2]), 'shear': float(params[3])}
            
        return best_parameters  
    

    def search_optimal_parameters_on_space(self, image_paths: List[Path], annotations: List[Annotation], target_embedding: np.ndarray):
        images = [Image.open(image_path) for image_path in image_paths]
        # Define the objective function for optimization
        @use_named_args(self.__generate_parameter_space())
        def objective(rotation, translate, scale, shear, brightness, contrast, saturation, hue):
            similarity_sum = 0
            for idx, (image_path, annotation) in enumerate(zip(image_paths, annotations)):
                image = images[idx]
                image_array = np.array(image)
                params = [rotation, translate, scale, shear, brightness, contrast, saturation, hue]
                transformation_result = self.__apply_transformation(image_array, annotation, params)
                if transformation_result:
                    transformed_image, transformed_bbox, class_index = transformation_result
                    x1, y1, x2, y2 = transformed_bbox
                    xc = (x1 + x2) / 2
                    yc = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    transformed_annotation = Annotation(
                        class_index,
                        Bbox(xc, yc, w, h).normalize(image.size),
                        confidence=annotation.confidence
                    )
                    embedding = self.produce_compressed_detection_embedding(transformed_image, transformed_annotation)
                    similarity = cosine_similarity(embedding, target_embedding)[0][0]
                    similarity_sum += similarity
                else:
                    similarity_sum += -1
            average_similarity = similarity_sum / len(annotations)
            # Minimize negative similarity for optimization
            return -average_similarity
        
        # Run Bayesian Optimization
        variable_space = self.__generate_parameter_space()
        res = gp_minimize(objective, variable_space, n_calls=60, random_state=0, verbose=True)

        [image.close() for image in images]
        # Extract the best parameters
        best_parameters = {
            variable.name : float(res.x[idx])
            for idx, variable in enumerate(variable_space)
        }
        # best_parameters = {
        #     'rotation': float(res.x[0]), 
        #     'translate': float(res.x[1]), 
        #     'scale': float(res.x[2]), 
        #     'shear': float(res.x[3])
        # }
        return best_parameters


    def produce_compressed_detection_embedding(
        self,
        image_path: Union[str, Path],
        transformed_annotation: Annotation, 
        conf_thres: float=0.47, # TODO must be a parameter
        iou_thres: float=0.7        
    ) -> np.ndarray:
        annotation_to_long_embedding = self.embedding_producer.get_image_embeddings(image_path, [transformed_annotation], conf_thres, iou_thres)
        assert(len(annotation_to_long_embedding) == 1)
        long_embedding = list(annotation_to_long_embedding.values())[0]
        compressed_embedding = self.pca_model.transform(long_embedding)
        return compressed_embedding # 1, 512 shape embedding
    
    @staticmethod
    def __generate_parameter_grid():
        """
        only parameters could be defined here! No extra code allowed!
        """
        # A.Affine
        rotations = np.linspace(-15, 16, 5)  # From -15 to +15 with a step of 1
        translations = np.linspace(-0.1, 0.1, 5)  # 10 points from -0.1 to 0.1
        scales = np.linspace(0.7, 1.3, 5)  # From 0.8 to 1.25 with a step of 0.1
        shears = np.linspace(-5, 6, 5)  # From -5 to +5 with a step of 1
        # A.ColorJitter
        # brightness = np.linspace(-0.2, 0.2, 5)
        # contrast = np.linspace(-0.2, 0.2, 5)
        # saturation = np.linspace(-0.5, 0.5, 5)
        # hue = np.linspace(-0.5, 0.5, 5)
        return np.array(np.meshgrid(rotations, translations, scales, shears)).T.reshape(-1, len(locals()))

    @staticmethod
    def __generate_parameter_space():
        # Define the search space for Bayesian Optimization
        return [
            Real(-15, 15, name='rotation'),
            Real(-0.1, 0.1, name='translate'),
            Real(0.7, 1.3, name='scale'),
            Real(-5, 5, name='shear'),
            Real(0, 0.3, name="brightness"),
            Real(0, 0.3, name="contrast"),
            Real(0, 0.6, name="saturation"),
            Real(0, 0.6, name="hue")
        ]

    @staticmethod
    def __apply_transformation(image: np.ndarray, annotation: Annotation, params):
        rotation, translate, scale, shear, brightness, contrast, saturation, hue = params
        transform = A.Compose([
            A.Affine(rotate=rotation, translate_percent={"x": translate, "y": translate}, scale=scale, shear=shear, always_apply=True),
            A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, always_apply=True)
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        
        # Note: The bbox must be in the format expected by bbox_params, here 'pascal_voc': [x_min, y_min, x_max, y_max]
        # category_ids is a list where each element corresponds to the class index of the bounding box at the same index.
        im_h, im_w, _ = image.shape
        bbox = annotation.bbox.to_absolute(image_wh=(im_w, im_h)).clip(image_wh=(im_w, im_h)).rectangle()
        transformed = transform(image=image, bboxes=[bbox], category_ids=[annotation.class_index])
        
        # Extracting the transformed image, bounding box, and class index
        transformed_image = transformed['image']
        try:
            transformed_bbox = transformed['bboxes'][0]  # Extract the first (and only) bounding box
        except IndexError:
            return None
        transformed_class_index = transformed['category_ids'][0]  # Extract the class index of the first (and only) bbox
        
        return transformed_image, transformed_bbox, transformed_class_index
    


if __name__ == "__main__":
    netron_layer_names = [
        '/model.15/cv2/act/Mul',
        '/model.22/Concat',
        '/model.22/Concat_1',
        '/model.22/Concat_2' 
    ]
    aug_opt = AugmentationOptimizer(
        embeddings_val_dir="/media/smarkov1001/storage_ssd/embedding_analysis_data/runs/E_FILTER/4_layers/REDUCED512/run64_on_val",
        embeddings_train_dir="/media/smarkov1001/storage_ssd/embedding_analysis_data/runs/E_FILTER/4_layers/REDUCED512/run64_train_from_GT",
        ground_truth_val_dir="/home/smarkov1001/sm_2x3080-1/outputs/human_head_detection_yolov8_dataset/val",
        ground_truth_train_dir="/home/smarkov1001/sm_2x3080-1/outputs/human_head_detection_yolov8_dataset/train",
        pca_model_path="/media/smarkov1001/storage_ssd/embedding_analysis_data/runs/E_FILTER/4_layers/REDUCED512_PCA_model.joblib",
        class_of_interest=1,
        onnx_model_path="/home/smarkov1001/NVI/scripts/models/run64.onnx",
        netron_layer_names=netron_layer_names
    )
    # aug_opt.find_n_nearest_train_images_to_each_val_fp(train_cluster_size=30, save=False)
    # aug_opt._test_copy_fp_and_corresponding_train_crops_to_separate_dir("/home/smarkov1001/NVI/scripts/humans/testing_case/")
    # aug_opt._find_augs()
    # aug_opt.find_best_augmentations_for_nearest_inputs_to_given_vals(
    #     train_cluster_size=30, 
    #     save_path="/home/smarkov1001/NVI/scripts/humans/testing_case/augs.yaml",
    #     mode='bayes'
    # )

    # aug_opt.assign_fp_coeff_to_each_train_embedding(
    #     output_yaml="/home/smarkov1001/NVI/scripts/TMP/similarity_pairs/similarity_data_sum_2nd_removal.yaml",
    #     mode='sum' # do not aggregate similarity values across FPs (i.e. keep list for each train embedding)
    # )
    aug_opt.visualize_find_n_nearest_train_images_to_each_val_fp(
        "/home/smarkov1001/NVI/scripts/TMP/similarity_pairs/vis_30_nearest",
        30, 
    )

    