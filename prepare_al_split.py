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

    conf_path = "/".join(weights.split("/")[:-2]) + "/best_conf.txt"
    with open(conf_path, "r") as f:
        conf = float(f.readline())

    txt_path = f"/ssd/temp/{dataset_name}/{from_split}"

    import time

    def check_busy():
        while len((embeds_names := glob.glob("/ssd/temp/*embeds*"))) > 0:
            if any([split_name in x for x in embeds_names]):
                break
            else:
                print("waiting 1 more minute, someone else is active learning too...")
                time.sleep(60)

    check_busy()

    with open(txt_path, "r") as f:
        lines = f.readlines()
        from_names = [os.path.basename(x)[:-1] for x in lines]
    print(
        "\n===============Populating image dir with corresponding formatted anno files...==============="
    )

    original_dataset = f"/ssd/temp/{dataset_name}/images/train/"
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

    embeds_dir = f"/ssd/temp/embeds_{from_fraction}_{split_name}"
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

        # ========================================
        netron_layer_names = [
            "/model.22/Concat",  # (1, 66, 48, 80)
            "/model.22/Concat_1",  # (1, 66, 24, 40)
            "/model.22/Concat_2",  # (1, 66, 12, 20)
        ]
        provider = [
            (
                "CUDAExecutionProvider",
                {"device_id": device},
            )
        ]
        yep = YoloEmbeddingsProducer(
            onnx_path, netron_layer_names, output_alias_names, providers=provider
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
    reduced_embeds_dir = f"/ssd/temp/reduced_embeds_{from_fraction}_{split_name}"

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

        print("Creating subset folder...")

        # copy small_amount of total files

        # small_amount = len(os.listdir(embeddings_source)) // 5
        # copy_names = random.sample(list(set([x[:-4] for x in os.listdir(embeddings_source)])), small_amount)
        # copy_names = set(copy_names)

        temp_folder = f"temp_folder_{from_fraction}_{split_name}"
        force_mkdir(temp_folder)

        # for file in tqdm(os.listdir(embeddings_source)):
        #     if file[:-4] in copy_names:
        #         shutil.copy(
        #             os.path.join(embeddings_source, file),
        #             os.path.join(temp_folder, file)
        #         )
        embeds_list = glob.glob(f"{embeddings_source}/*npy")
        for i, file in tqdm(enumerate(embeds_list)):
            if i % 5 == 0:
                base = file[:-4]
                for ext in [".npy", ".jpg", ".txt"]:
                    shutil.copy(
                        file[:-4] + ext,
                        os.path.join(temp_folder, os.path.basename(file))[:-4] + ext,
                    )
        # launch with same to_folder and small_amount from_folder
        print("Training PCA on a subset...")
        epp = EmbeddingPoolPreprocessor(
            temp_folder,
            reduced_embeds_dir,
            target_length=512,
            batch_size=512,
        )
        epp.run_dimension_reduction(mode="PCA")
        epp.synchronize_input_output_file_structure(
            embeddings_source, reduced_embeds_dir
        )

        # remove to_folder and small_amount folder
        shutil.rmtree(reduced_embeds_dir)
        shutil.rmtree(temp_folder)

        # and launch again with correct args
        epp = EmbeddingPoolPreprocessor(
            embeddings_source,
            reduced_embeds_dir,
            target_length=512,
            batch_size=512,
        )
        epp.run_dimension_reduction(mode="PCA")
        epp.synchronize_input_output_file_structure(
            embeddings_source, reduced_embeds_dir
        )

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
            first_list, second_list, k=target_num * (1 - bg2all_ratio), mode=mode
        )
        pickle_save(selected_path, selected)

    print(
        "\n===============Creating split with correct frg/bg proportion...==============="
    )
    not_bgs = [x + ".jpg" for x in selected]

    free_bgs = []
    for file in tqdm(
        glob.glob(f"/ssd/temp/{dataset_name}/labels/train/*txt", recursive=True)
    ):
        name = os.path.basename(file)
        if not os.path.getsize(file) and name not in from_names:
            free_bgs.append(file)

    bgs = [
        os.path.basename(x).replace("txt", "jpg")
        for x in random.sample(free_bgs, int(target_num * bg2all_ratio))
    ]
    res_path = f"/ssd/temp/{dataset_name}/train_{to_fraction}_{split_name}.txt"

    with open(res_path, "w") as f:
        f.writelines([f"./images/train/{x}\n" for x in from_names + not_bgs + bgs])

    print(f"`{res_path}` saved successfully.")

    yaml_path = f"{dataset_name}_{to_fraction}_{split_name}.yaml"
    with open(f"{dataset_name}.yaml", "r") as from_file:
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

        for file in glob.glob("/ssd/temp/*joblib"):
            os.remove(file)

        os.remove(first_list_path)
        os.remove(second_list_path)
        os.remove(selected_path)
        print("All cleaned up!")
