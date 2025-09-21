import argparse
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
    parser.add_argument("--netron-layer-names")
    parser.add_argument(
        "--index-backend", default="annoy", choices=["annoy", "hnsw"]
    )  # new flag
    parser.add_argument("--coarse-to-fine", action="store_true")  # new flag

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

    conf_path = "/".join(weights.split("/")[:-2]) + "/best_conf.txt"
    with open(conf_path, "r") as f:
        conf = float(f.readline())

    txt_path = f"/home/setupishe/datasets/{dataset_name}/{from_split}"

    import time

    def check_busy():
        while len((embeds_names := glob.glob("/home/setupishe/datasets/*embeds*"))) > 0:
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

    original_dataset = f"/home/setupishe/datasets/{dataset_name}/images/train/"
    if len(filelist := glob.glob(f"{original_dataset}*jpg")) == len(
        glob.glob(f"{original_dataset}*txt")
    ):
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

    onnx_path = weights.replace(".pt", ".onnx")

    if not os.path.exists(onnx_path):
        print("\n===============Converting weights to onnx===============")
        cmd = f"yolo export model={weights} format=onnx imgsz=640"
        os.system(cmd)

    print("\n===============Infering model on all available data...===============")

    embeds_dir = f"/home/setupishe/datasets/embeds_{from_fraction}_{split_name}"
    print("Estimating total embeds amount...")
    total_embeds_count = 0
    for file in tqdm(glob.glob(f"{original_dataset}*txt")):
        with open(file, "r") as f:
            total_embeds_count += len(f.readlines())
    compute_embeds = True
    if os.path.exists(embeds_dir):
        if (
            len(glob.glob(f"{embeds_dir}/**/*.npy", recursive=True))
            == total_embeds_count
        ):
            compute_embeds = False
            print("Skipping embeds computation since they already exist")
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
        )
        yep.produce_embeddings_for_dir(
            dir_path=original_dataset,
            embedding_and_crops_save_dir=embeds_dir,
            from_annotations_in_dir=True,
            conf_thres=float(conf),
            iou_thres=0.4,
            n=-1,
            random_images=False,
            each_k_th_image=None,
        )

    print("\n===============Preprocessing embeds with PCA...===============")
    embeddings_source = embeds_dir
    reduced_embeds_dir = (
        f"/home/setupishe/datasets/reduced_embeds_{from_fraction}_{split_name}"
    )

    preprocess_embeds = True
    if os.path.exists(reduced_embeds_dir):
        if (
            len(glob.glob(f"{reduced_embeds_dir}/**/*.npy", recursive=True))
            == total_embeds_count
        ):
            preprocess_embeds = False
            print("Skipping embeds preprocessing since its already done")
        else:
            shutil.rmtree(reduced_embeds_dir)

    if preprocess_embeds:
        if skip_pca:
            print("Skipping PCA, running normalization only...")
            epp = EmbeddingPoolPreprocessor(
                embeddings_source,
                reduced_embeds_dir,
                batch_size=512,
            )
            epp.run_l2_normalization()
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

    # #removing whole img embeds, leaving only bboxes embeds
    # for file in glob.glob(f"{reduced_embeds_dir}/**/*npy, recursive=True"):
    #     if len(os.path.basename(file).split("_")) == 2:
    #         os.remove(file)

    print("\n===============Selecting samples===============")
    target_num = len(filelist) * (to_fraction - from_fraction)
    first_list_path = f"first_list_{from_fraction}_{split_name}.pickle"
    second_list_path = f"second_list_{from_fraction}_{split_name}.pickle"

    print("Creating filelists for embeds selector...")
    if all([os.path.exists(item) for item in [first_list_path, second_list_path]]):
        first_list = pickle_load(first_list_path)
        second_list = pickle_load(second_list_path)
        print("Skipping, lists already exist...")
    else:
        first_list = []
        second_list = []
        basenames = [x[:-4] for x in from_names]
        for file in tqdm(glob.glob(f"{reduced_embeds_dir}/**/*npy", recursive=True)):
            file_base_name = os.path.basename(file)
            img_name = file_base_name[: file_base_name.index("_cropped")]
            if img_name in basenames:
                first_list.append(file)
            else:
                second_list.append(file)

        pickle_save(first_list_path, first_list)
        pickle_save(second_list_path, second_list)
    selected_path = f"selected_embeds_{from_fraction}_{split_name}.pickle"
    if os.path.exists(selected_path):
        selected = pickle_load(selected_path)
        print("Skipping, selected embeds already exist...")
    else:
        selected = select_embeddings(
            first_list,
            second_list,
            k=target_num * (1 - bg2all_ratio),
            mode=mode,
            backend=index_backend,
            coarse_to_fine=coarse_to_fine,
        )
        pickle_save(selected_path, selected)

    print(
        "\n===============Creating split with correct frg/bg proportion...==============="
    )
    not_bgs = [x + ".jpg" for x in selected]

    free_bgs = []
    for file in tqdm(
        glob.glob(
            f"/home/setupishe/datasets/{dataset_name}/labels/train/*txt", recursive=True
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
        f"/home/setupishe/datasets/{dataset_name}/train_{to_fraction}_{split_name}.txt"
    )

    with open(res_path, "w") as f:
        f.writelines([f"./images/train/{x}\n" for x in from_names + not_bgs + bgs])

    print(f"`{res_path}` saved successfully.")

    yaml_path = f"/home/setupishe/ultralytics/ultralytics/cfg/datasets/{dataset_name}_{to_fraction}_{split_name}.yaml"
    with open(
        f"/home/setupishe/ultralytics/ultralytics/cfg/datasets/{dataset_name}.yaml", "r"
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

        for file in glob.glob(f"{original_dataset}*txt"):
            os.remove(file)

        shutil.rmtree(embeds_dir)
        shutil.rmtree(reduced_embeds_dir)

        for file in glob.glob("/home/setupishe/datasets/*joblib"):
            os.remove(file)

        os.remove(first_list_path)
        os.remove(second_list_path)
        os.remove(selected_path)
        print("All cleaned up!")
