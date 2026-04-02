#!/bin/bash

DESTINATION_DIR="../models/baked_world/meshes"
FILE_NAME="terrain_with_vegetation.glb"
DRIVE_FILE_ID="1z7y6pbHiJ_kHUXa3plmh0uwXIwP2N1sH"

echo "Checking for the 2.3GB Desert Mesh..."

if [ -f "$DESTINATION_DIR/$FILE_NAME" ]; then
    echo "Mesh already exists. Skipping download."
else
    echo "Mesh missing. Fetching from Google Drive..."
    
    if ! command -v gdown &> /dev/null; then
        echo "Installing gdown..."
        pip install gdown
    fi

    mkdir -p $DESTINATION_DIR
    gdown --id $DRIVE_FILE_ID -O $DESTINATION_DIR/$FILE_NAME

    echo "Download complete! Ready for Gazebo."
fi