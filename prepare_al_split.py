import argparse
import json
from al_utils import *
from produce_detection_embeddings import *
from preprocess_embedding_pool import *

if __name__ == "__main__":
    print("\n===============Initializing...===============")
    parser = argparse.ArgumentParser()

    parser.add_argument("--from-fraction")
    parser.add_argument("--to-fraction")
    parser.add_argument("--from-split")
    parser.add_argument("--weights")
    parser.add_argument("--device")
    parser.add_argument("--split-name")
    parser.add_argument("--dataset-name")
    parser.add_argument("--bg2all-ratio")
    parser.add_argument("--mode", default="distance")
    parser.add_argument("--cleanup", action="store_true")  # on/off flag
    parser.add_argument("--seg2line", action="store_true")  # on/off flag
    parser.add_argument("--skip-pca", action="store_true")  # on/off flag
    parser.add_argument(
        "--from-predictions",
        action="store_true",
        help="Use model predictions as bbox source (ignore per-image .txt annotation files). Default: use per-image .txt annotations.",
    )
    parser.add_argument("--netron-layer-names")
    parser.add_argument(
        "--separate-maps-voting",
        action="store_true",
        help="Store 3 feature-map embeddings separately and select per-map independently, then combine selected images by voting (1..3).",
    )
    parser.add_argument(
        "--roi-hw",
        "--embedding-hw",
        dest="embedding_hw",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Override ROI Align output resolution (H W) used before flattening embeddings. Default keeps current behavior.",
    )
    parser.add_argument(
        "--index-backend", default="annoy", choices=["annoy", "hnsw"]
    )  # new flag
    parser.add_argument(
        "--datasets-dir",
        default="/home/setupishe/datasets",
        help="Base directory that contains all datasets (default: /home/setupishe/datasets).",
    )
    parser.add_argument(
        "--ultralytics-cfg-dir",
        default="/home/setupishe/ultralytics/ultralytics/cfg/datasets",
        help="Directory containing ultralytics YAML dataset configs (default: /home/setupishe/ultralytics/ultralytics/cfg/datasets).",
    )
    parser.add_argument("--coarse-to-fine", action="store_true")
    parser.add_argument(
        "--granularity-divs",
        default="8 4 2 1",
        help="Space-separated divisors for matryoshka_variance granularity levels. "
             "Each value d produces prefix dim = embedding_dim // d (default: '8 4 2 1').",
    )
    parser.add_argument(
        "--ctf-k1-mult",
        type=float,
        default=4.0,
        help="Coarse-to-fine Stage-1 candidate multiplier. k1 = k * ctf_k1_mult (default: 4).",
    )
    parser.add_argument(
        "--ctf-k2-mult",
        type=float,
        default=2.0,
        help="Coarse-to-fine Stage-2 candidate multiplier. k2 = k * ctf_k2_mult (default: 2).",
    )
    parser.add_argument(
        "--ctf-d1-div",
        type=int,
        default=8,
        help="Coarse-to-fine Stage-1 dimension divisor. d1 = embedding_dim // ctf_d1_div (default: 8).",
    )
    parser.add_argument(
        "--ctf-d2-div",
        type=int,
        default=4,
        help="Coarse-to-fine Stage-2 dimension divisor. d2 = embedding_dim // ctf_d2_div (default: 4).",
    )

    args = parser.parse_args()

    from_fraction = float(args.from_fraction)
    to_fraction = float(args.to_fraction)
    from_split = args.from_split
    weights = args.weights
    device = args.device
    bg2all_ratio = float(args.bg2all_ratio)
    mode = args.mode
    dataset_name = args.dataset_name
    split_name = args.split_name
    cleanup = args.cleanup
    seg2line = args.seg2line
    skip_pca = args.skip_pca
    index_backend = args.index_backend
    coarse_to_fine = args.coarse_to_fine
    granularity_divs = [int(x) for x in args.granularity_divs.split()]
    ctf_k1_mult = args.ctf_k1_mult
    ctf_k2_mult = args.ctf_k2_mult
    ctf_d1_div = args.ctf_d1_div
    ctf_d2_div = args.ctf_d2_div
    embedding_hw = args.embedding_hw
    separate_maps_voting = args.separate_maps_voting
    datasets_dir = args.datasets_dir
    ultralytics_cfg_dir = args.ultralytics_cfg_dir
    # Default bbox source is annotations (historical behavior): `from_annotations_in_dir=True`.
    # If user passes `--from-predictions`, switch to predicted bboxes.
    from_annotations_in_dir = not args.from_predictions

    def _done_path(folder: str) -> str:
        return os.path.join(folder, "_DONE.json")

    def _write_done(folder: str, payload: dict) -> None:
        os.makedirs(folder, exist_ok=True)
        with open(_done_path(folder), "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _count_npy(folder: str) -> int:
        return len(glob.glob(f"{folder}/**/*.npy", recursive=True))

    conf_path = "/".join(weights.split("/")[:-2]) + "/best_conf.txt"
    with open(conf_path, "r") as f:
        conf = float(f.readline())

    txt_path = f"{datasets_dir}/{dataset_name}/{from_split}"

    import time

    def check_busy():
        while len((embeds_names := glob.glob(f"{datasets_dir}/*embeds*"))) > 0:
            if any([split_name in x for x in embeds_names]):
                break
            else:
                print(embeds_names)
                print("waiting 1 more minute, someone else is active learning too...")
                time.sleep(60)

    check_busy()

    with open(txt_path, "r") as f:
        lines = f.readlines()
        from_names = [os.path.basename(x)[:-1] for x in lines]
    print(
        "\n===============Populating image dir with corresponding formatted anno files...==============="
    )

    original_dataset = f"{datasets_dir}/{dataset_name}/images/train/"
    filelist = glob.glob(f"{original_dataset}*jpg")
    if from_annotations_in_dir:
        if len(filelist) == len(glob.glob(f"{original_dataset}*txt")):
            print("Skipping populating, already exist...")
        else:
            for file in tqdm(filelist):
                label_file = file.replace("images", "labels").replace("jpg", "txt")
                if os.path.exists(label_file):
                    segfile2bboxfile(
                        label_file, file.replace("jpg", "txt"), seg2line=seg2line
                    )
                else:
                    os.mknod(file.replace("jpg", "txt"))
    else:
        print("Skipping populating annotation txts because --from-predictions is enabled.")

    onnx_path = weights.replace(".pt", ".onnx")

    if not os.path.exists(onnx_path):
        print("\n===============Converting weights to onnx===============")
        cmd = f"yolo export model={weights} format=onnx imgsz=640"
        os.system(cmd)

    print("\n===============Infering model on all available data...===============")

    embeds_dir = f"{datasets_dir}/embeds_{from_fraction}_{split_name}"
    print("Estimating total embeds amount...")
    expected_embeddings_files = None
    if from_annotations_in_dir:
        total_embeds_count = 0
        for file in tqdm(glob.glob(f"{original_dataset}*txt")):
            with open(file, "r") as f:
                total_embeds_count += len(f.readlines())
        expected_embeddings_files = (
            total_embeds_count * 3 if separate_maps_voting else total_embeds_count
        )
    else:
        print("Using predictions as bbox source; embeds count will be derived from produced .npy files.")
    compute_embeds = True
    if os.path.exists(embeds_dir):
        done_exists = os.path.exists(_done_path(embeds_dir))
        actual_npy = _count_npy(embeds_dir)
        if done_exists and (
            (expected_embeddings_files is None) or (actual_npy == expected_embeddings_files)
        ):
            compute_embeds = False
            print("Skipping embeds computation since they already exist (_DONE.json present)")
        else:
            shutil.rmtree(embeds_dir)

    if compute_embeds:
        output_alias_names = {}

        default_netron_layer_names = [
            "/model.22/Concat",
            "/model.22/Concat_1",
            "/model.22/Concat_2",
        ]
        # ========================================
        netron_layer_names = (
            args.netron_layer_names.split()
            if "netron_layer_names" in args
            else default_netron_layer_names
        )
        if separate_maps_voting and len(netron_layer_names) != 3:
            raise ValueError(
                f"--separate-maps-voting expects exactly 3 feature maps (got {len(netron_layer_names)}): {netron_layer_names}"
            )
        if separate_maps_voting:
            strategy = "separate"
        else:
            strategy = (
                "default"
                if args.netron_layer_names == default_netron_layer_names
                else "iter"
            )
        provider = [
            (
                "CUDAExecutionProvider",
                {"device_id": device},
            )
        ]
        yep = YoloEmbeddingsProducer(
            onnx_path,
            netron_layer_names,
            output_alias_names,
            providers=provider,
            strategy=strategy,
            embedding_tensors_hw_resolution_before_flattening=embedding_hw,
        )
        yep.produce_embeddings_for_dir(
            dir_path=original_dataset,
            embedding_and_crops_save_dir=embeds_dir,
            from_annotations_in_dir=from_annotations_in_dir,
            conf_thres=float(conf),
            iou_thres=0.4,
            n=-1,
            random_images=False,
            each_k_th_image=None,
        )
        # Mark completion and record actual count (works for both annotation- and prediction-based bbox sources).
        actual_npy = _count_npy(embeds_dir)
        expected_embeddings_files = actual_npy
        _write_done(
            embeds_dir,
            {
                "stage": "embeddings",
                "bbox_source": "annotations" if from_annotations_in_dir else "predictions",
                "separate_maps_voting": bool(separate_maps_voting),
                "netron_layer_names": netron_layer_names,
                "embedding_hw": embedding_hw,
                "onnx_path": onnx_path,
                "conf": float(conf),
                "iou_thres": 0.4,
                "npy_files": int(actual_npy),
            },
        )
    else:
        # If we skipped computation, still define expected_embeddings_files for downstream checks.
        if expected_embeddings_files is None:
            expected_embeddings_files = _count_npy(embeds_dir)

    print("\n===============Preprocessing embeds with PCA...===============")
    embeddings_source = embeds_dir
    reduced_embeds_dir = (
        f"{datasets_dir}/reduced_embeds_{from_fraction}_{split_name}"
    )

    preprocess_embeds = True
    if os.path.exists(reduced_embeds_dir):
        done_exists = os.path.exists(_done_path(reduced_embeds_dir))
        actual_npy = _count_npy(reduced_embeds_dir)
        if done_exists and (
            (expected_embeddings_files is None) or (actual_npy == expected_embeddings_files)
        ):
            preprocess_embeds = False
            print("Skipping embeds preprocessing since its already done (_DONE.json present)")
        else:
            shutil.rmtree(reduced_embeds_dir)

    if preprocess_embeds:
        if skip_pca:
            print("Skipping PCA, running normalization only...")
            if separate_maps_voting:
                # Important: separate maps have different embedding lengths, so we must normalize
                # each map independently (otherwise torch stack will fail in the DataLoader).
                for mi in range(3):
                    files = glob.glob(
                        f"{embeddings_source}/**/*.m{mi}.npy", recursive=True
                    )
                    if len(files) == 0:
                        raise RuntimeError(
                            f"--separate-maps-voting enabled, but no .m{mi}.npy files found in {embeddings_source}"
                        )
                    epp = EmbeddingPoolPreprocessor(
                        files,
                        reduced_embeds_dir,
                        batch_size=512,
                    )
                    epp.run_l2_normalization()
            else:
                epp = EmbeddingPoolPreprocessor(
                    embeddings_source,
                    reduced_embeds_dir,
                    batch_size=512,
                )
                epp.run_l2_normalization()
            _write_done(
                reduced_embeds_dir,
                {
                    "stage": "preprocess",
                    "mode": "l2_normalization_only",
                    "separate_maps_voting": bool(separate_maps_voting),
                    "source_dir": embeddings_source,
                    "npy_files": int(_count_npy(reduced_embeds_dir)),
                },
            )
        else:
            print("Creating subset folder for PCA training...")
            temp_folder = f"temp_folder_{from_fraction}_{split_name}"
            force_mkdir(temp_folder)
            embeds_list = glob.glob(f"{embeddings_source}/*npy")
            for i, file in tqdm(enumerate(embeds_list)):
                if i % 5 == 0:
                    base = file[:-4]
                    for ext in [".npy", ".jpg", ".txt"]:
                        if os.path.exists(file[:-4] + ext):
                            shutil.copy(
                                file[:-4] + ext,
                                os.path.join(temp_folder, os.path.basename(file))[:-4]
                                + ext,
                            )

            print("Training PCA on a subset...")
            epp = EmbeddingPoolPreprocessor(
                temp_folder,
                reduced_embeds_dir,
                target_length=512,
                batch_size=512,
            )
            epp.run_dimension_reduction(mode="PCA")

            print("Applying trained PCA to all embeddings...")
            epp = EmbeddingPoolPreprocessor(
                embeddings_source,
                reduced_embeds_dir,
                target_length=512,
                batch_size=512,
            )
            epp.run_dimension_reduction(mode="PCA")

            shutil.rmtree(temp_folder)
            _write_done(
                reduced_embeds_dir,
                {
                    "stage": "preprocess",
                    "mode": "pca_then_l2",
                    "separate_maps_voting": bool(separate_maps_voting),
                    "source_dir": embeddings_source,
                    "target_length": 512,
                    "npy_files": int(_count_npy(reduced_embeds_dir)),
                },
            )

    # #removing whole img embeds, leaving only bboxes embeds
    # for file in glob.glob(f"{reduced_embeds_dir}/**/*npy, recursive=True"):
    #     if len(os.path.basename(file).split("_")) == 2:
    #         os.remove(file)

    print("\n===============Selecting samples===============")
    target_num = len(filelist) * (to_fraction - from_fraction)
    list_suffix = "_separate_vote" if separate_maps_voting else ""
    first_list_path = f"first_list_{from_fraction}_{split_name}{list_suffix}.pickle"
    second_list_path = f"second_list_{from_fraction}_{split_name}{list_suffix}.pickle"

    print("Creating filelists for embeds selector...")
    if all([os.path.exists(item) for item in [first_list_path, second_list_path]]):
        filelists = pickle_load(first_list_path), pickle_load(second_list_path)
        print("Skipping, lists already exist...")
    else:
        basenames = [x[:-4] for x in from_names]
        if separate_maps_voting:
            first_lists = [[], [], []]
            second_lists = [[], [], []]
            for file in tqdm(
                glob.glob(f"{reduced_embeds_dir}/**/*npy", recursive=True)
            ):
                base = os.path.basename(file)
                # Expect naming from producer: xxx_cropped.m0.npy / .m1.npy / .m2.npy
                if ".m0.npy" in base:
                    mi = 0
                elif ".m1.npy" in base:
                    mi = 1
                elif ".m2.npy" in base:
                    mi = 2
                else:
                    continue
                img_name = base[: base.index("_cropped")]
                if img_name in basenames:
                    first_lists[mi].append(file)
                else:
                    second_lists[mi].append(file)
            pickle_save(first_list_path, first_lists)
            pickle_save(second_list_path, second_lists)
            filelists = (first_lists, second_lists)
        else:
            first_list = []
            second_list = []
            for file in tqdm(
                glob.glob(f"{reduced_embeds_dir}/**/*npy", recursive=True)
            ):
                file_base_name = os.path.basename(file)
                img_name = file_base_name[: file_base_name.index("_cropped")]
                if img_name in basenames:
                    first_list.append(file)
                else:
                    second_list.append(file)
            pickle_save(first_list_path, first_list)
            pickle_save(second_list_path, second_list)
            filelists = (first_list, second_list)

    selected_path = f"selected_embeds_{from_fraction}_{split_name}{list_suffix}.pickle"
    if os.path.exists(selected_path):
        selected = pickle_load(selected_path)
        print("Skipping, selected embeds already exist...")
    else:
        k_sel = int(target_num * (1 - bg2all_ratio))
        if separate_maps_voting:
            first_lists, second_lists = filelists
            selected = select_embeddings_voting(
                first_lists,
                second_lists,
                k=k_sel,
                mode=mode,
                backend=index_backend,
                coarse_to_fine=coarse_to_fine,
                coarse_k1_mult=ctf_k1_mult,
                coarse_k2_mult=ctf_k2_mult,
                coarse_d1_div=ctf_d1_div,
                coarse_d2_div=ctf_d2_div,
            )
        else:
            first_list, second_list = filelists
            selected = select_embeddings(
                first_list,
                second_list,
                k=k_sel,
                mode=mode,
                backend=index_backend,
                coarse_to_fine=coarse_to_fine,
                coarse_k1_mult=ctf_k1_mult,
                coarse_k2_mult=ctf_k2_mult,
                coarse_d1_div=ctf_d1_div,
                coarse_d2_div=ctf_d2_div,
                granularity_divs=granularity_divs,
            )
        pickle_save(selected_path, selected)

    print(
        "\n===============Creating split with correct frg/bg proportion...==============="
    )
    not_bgs = [x + ".jpg" for x in selected]

    free_bgs = []
    for file in tqdm(
        glob.glob(
            f"{datasets_dir}/{dataset_name}/labels/train/*txt", recursive=True
        )
    ):
        name = os.path.basename(file)
        if not os.path.getsize(file) and name not in from_names:
            free_bgs.append(file)

    bgs = [
        os.path.basename(x).replace("txt", "jpg")
        for x in random.sample(free_bgs, int(target_num * bg2all_ratio))
    ]
    res_path = (
        f"{datasets_dir}/{dataset_name}/train_{to_fraction}_{split_name}.txt"
    )

    with open(res_path, "w") as f:
        f.writelines([f"./images/train/{x}\n" for x in from_names + not_bgs + bgs])

    print(f"`{res_path}` saved successfully.")

    yaml_path = f"{ultralytics_cfg_dir}/{dataset_name}_{to_fraction}_{split_name}.yaml"
    with open(
        f"{ultralytics_cfg_dir}/{dataset_name}.yaml", "r"
    ) as from_file:
        lines = from_file.readlines()

    for i, line in enumerate(lines):
        if "train: train.txt" in line:
            lines[i] = lines[i].replace(
                "train.txt", f"train_{to_fraction}_{split_name}.txt"
            )
    with open(
        yaml_path,
        "w",
    ) as to_file:
        to_file.writelines(lines)
    print(f"`{yaml_path}` saved successfully.")

    if cleanup:
        print("\n===============Cleaning up...===============")

        # Only remove per-image annotation txts if we created/needed them.
        if from_annotations_in_dir:
            for file in glob.glob(f"{original_dataset}*txt"):
                os.remove(file)

        shutil.rmtree(embeds_dir)
        shutil.rmtree(reduced_embeds_dir)

        for file in glob.glob(f"{datasets_dir}/*joblib"):
            os.remove(file)

        os.remove(first_list_path)
        os.remove(second_list_path)
        os.remove(selected_path)
        print("All cleaned up!")
