#!/bin/bash

# Unified Active Learning Chain Runner
# Usage: bash unified_chain_run.sh <config_file.json>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file.json>"
    echo "Available configs:"
    echo "  - configs/density_config.json"
    echo "  - configs/distance_config.json"
    echo "  - configs/confidence_config.json"
    echo "  - configs/distance_config_tune.json"
    exit 1
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Check if jq is installed for JSON parsing
if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required for JSON parsing. Please install it first."
    exit 1
fi

# Parse configuration
readarray -t RANGES < <(jq -r '.ranges[]' "$CONFIG_FILE")
DEVICE=$(jq -r '.device' "$CONFIG_FILE")
MODE=$(jq -r '.mode' "$CONFIG_FILE")
DATASET_NAME=$(jq -r '.dataset_name' "$CONFIG_FILE")
BG2ALL_RATIO=$(jq -r '.bg2all_ratio' "$CONFIG_FILE")
PREPARE_SCRIPT=$(jq -r '.prepare_script' "$CONFIG_FILE")
WEIGHTS_BASE_PATH=$(jq -r '.weights_base_path' "$CONFIG_FILE")
TUNE=$(jq -r '.tune // "false"' "$CONFIG_FILE")

# Export common vars for templating inside yolo args
export MODE
export DATASET_NAME
export DEVICE

echo "=== Active Learning Chain Runner ==="
echo "Config: $CONFIG_FILE"
echo "Mode: $MODE"
echo "Dataset: $DATASET_NAME"
echo "Device: $DEVICE"
echo "Ranges: ${RANGES[*]}"
echo "Tune mode: $TUNE"
echo "================================"

# Loop through each range value
for range in "${RANGES[@]}"; do
    # Calculate the next range value using bc for floating-point arithmetic
    next_range=$(printf "%.1f" $(echo "$range + 0.1" | bc))

    # Determine folder/fromsplit names
    if [[ "$range" == "0.2" ]]; then
        folder_name="random"
        fromsplit_name=""
    else
        folder_name="$MODE"
        fromsplit_name="_$MODE"
    fi

    echo "PREPARING ON FRACTION $range FOR FRACTION $next_range"

    PREV_MODEL_WEIGHTS="$WEIGHTS_BASE_PATH/VOC_${folder_name}_${range}/weights/best.pt"

    if [ "$PREPARE_SCRIPT" = "conf_criteria.py" ]; then
        python3 conf_criteria.py \
            --weights "$PREV_MODEL_WEIGHTS" \
            --from-fraction $range \
            --to-fraction $next_range \
            --from-split "train_${range}.txt" \
            --dataset-name "$DATASET_NAME" \
            --default-split train.txt \
            --split-name "$MODE" \
            --bg2all-ratio $BG2ALL_RATIO \
            --cleanup
    else
        python3 prepare_al_split.py \
            --weights "$PREV_MODEL_WEIGHTS" \
            --from-fraction $range \
            --to-fraction $next_range \
            --from-split "train_${range}${fromsplit_name}.txt" \
            --dataset-name "$DATASET_NAME" \
            --split-name "$MODE" \
            --mode "$MODE" \
            --bg2all-ratio $BG2ALL_RATIO \
            --device $DEVICE \
            --cleanup
    fi

    echo "TRAINING ON FRACTION $next_range"

    # Build YOLO args from JSON block .yolo_args (supports arbitrary args)
    # Allows templating with ${MODE}, ${DATASET_NAME}, ${next_range}, ${DEVICE}
    YOLO_ARGS=()
    # Define a function to add or update a key in YOLO_ARGS
    add_or_update_yolo_arg() {
        local key_to_check="$1"
        local new_value="$2"
        local found=0
        # Check if the key already exists and update it
        for i in "${!YOLO_ARGS[@]}"; do
            if [[ "${YOLO_ARGS[$i]}" == "$key_to_check="* ]]; then
                YOLO_ARGS[$i]="${key_to_check}=${new_value}"
                found=1
                break
            fi
        done
        # If the key was not found, add it
        if [ $found -eq 0 ]; then
            YOLO_ARGS+=("${key_to_check}=${new_value}")
        fi
    }


    while IFS= read -r key; do
        raw_val=$(jq -r --arg k "$key" '.yolo_args[$k]' "$CONFIG_FILE")
        # Expand placeholders
        export next_range
        val=$(eval echo "$raw_val")
        YOLO_ARGS+=("$key=$val")
    done < <(jq -r '.yolo_args | keys[]' "$CONFIG_FILE")

    if [[ "$TUNE" == "true" ]]; then
        echo "Fine-tuning mode enabled. Adjusting hyperparameters."

        # Set model to previous weights for fine-tuning, but not for the first iteration
        if [[ "$range" != "${RANGES[0]}" ]] || [ -f "$PREV_MODEL_WEIGHTS" ]; then
            add_or_update_yolo_arg "model" "$PREV_MODEL_WEIGHTS"
        fi

        # Get base hyperparameters from config or set defaults
        BASE_EPOCHS=$(jq -r '.yolo_args.epochs // 65' "$CONFIG_FILE")
        BASE_LR0=$(jq -r '.yolo_args.lr0 // 0.01' "$CONFIG_FILE")
        BASE_CLOSE_MOSAIC=$(jq -r '.yolo_args.close_mosaic // 10' "$CONFIG_FILE")

        # Calculate relative dataset increase
        RELATIVE_INCREASE=$(echo "($next_range - $range) / $range" | bc -l)

        # Adjust hyperparameters based on relative increase
        NEW_EPOCHS=$(printf "%.0f" $(echo "$BASE_EPOCHS * $RELATIVE_INCREASE" | bc -l))
        NEW_LR0=$(echo "$BASE_LR0 * $RELATIVE_INCREASE" | bc -l)
        NEW_CLOSE_MOSAIC=$(printf "%.0f" $(echo "$BASE_CLOSE_MOSAIC * $RELATIVE_INCREASE" | bc -l))

        echo "Relative dataset increase: $RELATIVE_INCREASE"
        echo "Adjusted epochs: $NEW_EPOCHS (base: $BASE_EPOCHS)"
        echo "Adjusted lr0: $NEW_LR0 (base: $BASE_LR0)"
        echo "Adjusted close_mosaic: $NEW_CLOSE_MOSAIC (base: $BASE_CLOSE_MOSAIC)"

        # Override YOLO_ARGS with new values
        add_or_update_yolo_arg "epochs" "$NEW_EPOCHS"
        add_or_update_yolo_arg "lr0" "$NEW_LR0"
        add_or_update_yolo_arg "close_mosaic" "$NEW_CLOSE_MOSAIC"
    fi

    # Run YOLO
    yolo train "${YOLO_ARGS[@]}"

    echo "Completed fraction $next_range"
    echo "------------------------"
done

echo "=== Active Learning Chain Complete ===" 