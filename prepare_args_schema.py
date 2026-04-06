PREPARE_BOOL_KEYS = frozenset(
    {
        "batched_inference",
        "save_crops",
        "seg2line",
        "cleanup",
        "skip_pca",
        "use_standard_scaler",
        "coarse_to_fine",
        "from_predictions",
        "separate_maps_voting",
        "time",
        "pseudo_mean_conf",
    }
)

PREPARE_VALUE_KEYS = frozenset(
    {
        "index_backend",
        "io_workers",
        "onnx_batch_size",
        "granularity_divs",
        "ctf_k1_mult",
        "ctf_k2_mult",
        "ctf_d1_div",
        "ctf_d2_div",
        "netron_layer_names",
        "seed",
        "image_aggregation",
        "use_dim",
        "train_subdir",
        "rgc_batch_size",
        "rgc_baseline_split",
    }
)

PREPARE_KNOWN_KEYS = PREPARE_BOOL_KEYS | PREPARE_VALUE_KEYS | frozenset({"roi_hw"})
CONF_CRITERIA_PREPARE_KEYS = frozenset({"seg2line", "cleanup", "pseudo_mean_conf"})
