#!/bin/bash
set -e

echo "============================================================"
echo "          8xH100 Bare-Metal Setup Script (PhD)              "
echo "============================================================"
echo ""

# 1. Prompt for dataset directory
read -p "Enter the absolute path to the pre-transferred datasets (VOC/COCO) directory: " DATASETS_DIR

if [ ! -d "$DATASETS_DIR" ]; then
    echo "Error: Directory '$DATASETS_DIR' does not exist."
    exit 1
fi

echo "Dataset directory set to: $DATASETS_DIR"
echo ""

# 2. Setup Conda environment
echo "Creating Conda environment 'phd_env' with Python 3.10..."
conda create -n phd_env python=3.10 -y

# Activate conda environment (requires conda to be initialized in the shell)
eval "$(conda shell.bash hook)"
conda activate phd_env

# 3. Install PyTorch and dependencies
echo "Installing PyTorch and dependencies..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install pandas numpy matplotlib seaborn pyyaml tqdm

# 4. Install ultralytics fork in editable mode
echo "Installing ultralytics fork in editable mode..."
# Assuming the script is run from bel_conf and ultralytics is a sibling directory
ULTRALYTICS_DIR="$(cd "$(dirname "$0")/../ultralytics" && pwd)"
if [ -d "$ULTRALYTICS_DIR" ]; then
    cd "$ULTRALYTICS_DIR"
    pip install -e .
    cd - > /dev/null
else
    echo "Warning: ultralytics directory not found at $ULTRALYTICS_DIR. Please install manually."
fi

# 5. Update Ultralytics settings
echo "Updating Ultralytics settings.yaml..."
SETTINGS_FILE="$HOME/.config/Ultralytics/settings.yaml"
mkdir -p "$(dirname "$SETTINGS_FILE")"

# Create a default settings file if it doesn't exist, or update the existing one
if [ ! -f "$SETTINGS_FILE" ]; then
    echo "datasets_dir: $DATASETS_DIR" > "$SETTINGS_FILE"
else
    # Use sed to replace the datasets_dir line, or append it if it doesn't exist
    if grep -q "^datasets_dir:" "$SETTINGS_FILE"; then
        sed -i "s|^datasets_dir:.*|datasets_dir: $DATASETS_DIR|" "$SETTINGS_FILE"
    else
        echo "datasets_dir: $DATASETS_DIR" >> "$SETTINGS_FILE"
    fi
fi

echo "Updated $SETTINGS_FILE with datasets_dir: $DATASETS_DIR"

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "To activate the environment, run: conda activate phd_env"
echo "Please refer to README_8xH100.md for GPU and dataset instructions."
echo "============================================================"
