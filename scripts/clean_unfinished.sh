#!/bin/bash

set -e

MODELS_DIR="./models"
RUNS_DIR="./runs"

# Safety checks
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory '$MODELS_DIR' does not exist."
    exit 1
fi

echo "Scanning for folders without FINISHED file..."
echo ""

# Arrays to store paths for deletion
models_to_delete=()
runs_to_delete=()

# Find all model directories without FINISHED file
while read -r model_dir; do
    if [ ! -f "$model_dir/FINISHED" ]; then
        dir_name=$(basename "$model_dir")
        models_to_delete+=("$model_dir")

        # Check for corresponding runs folder
        runs_folder="$RUNS_DIR/$dir_name"
        if [ -d "$runs_folder" ]; then
            runs_to_delete+=("$runs_folder")
        fi
    fi
done < <(find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d)

echo "${models_to_delete[@]}"

# Display what would be deleted
if [ ${#models_to_delete[@]} -eq 0 ] && [ ${#runs_to_delete[@]} -eq 0 ]; then
    echo "No folders to delete - all model folders have FINISHED files."
    exit 0
fi

echo "The following folders will be DELETED:"
echo ""

if [ ${#models_to_delete[@]} -gt 0 ]; then
    echo "MODELS folders to delete:"
    for model in "${models_to_delete[@]}"; do
        echo "  - $model"
    done
    echo ""
fi

if [ ${#runs_to_delete[@]} -gt 0 ]; then
    echo "RUNS folders to delete:"
    for run in "${runs_to_delete[@]}"; do
        echo "  - $run"
    done
    echo ""
fi

echo "Total folders to delete: $((${#models_to_delete[@]} + ${#runs_to_delete[@]}))"
echo ""

# Confirmation prompt
read -p "Are you sure you want to delete these folders? (y/N): " -n 1 -r
echo ""
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deletion cancelled."
    exit 0
fi

# Actually delete the folders
echo "Deleting folders..."
echo ""

# Delete models folders
for model in "${models_to_delete[@]}"; do
    echo "Deleting model folder: $model"
    rm -rf "$model"
done

# Delete runs folders
for run in "${runs_to_delete[@]}"; do
    echo "Deleting runs folder: $run"
    rm -rf "$run"
done

echo ""
echo "Deletion completed."