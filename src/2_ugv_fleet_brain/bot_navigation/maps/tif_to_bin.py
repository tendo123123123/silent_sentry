#!/usr/bin/env python3
import os
import sys
import struct
import numpy as np
from PIL import Image

def convert_tif_to_bin(tif_path, bin_path):
    if not os.path.exists(tif_path):
        print(f"Error: TIFF file not found at {tif_path}")
        sys.exit(1)
        
    print(f"Reading TIFF: {tif_path}")
    img = Image.open(tif_path)
    data = np.array(img, dtype=np.float32)
    rows, cols = data.shape
    print(f"Loaded image size: {cols}x{rows} ({data.size} pixels)")
    
    # Eigen MatrixXf is ColMajor by default.
    # To write the data in column-major order, we use order='F' (Fortran/column-major contiguous)
    col_major_data = data.astype(np.float32).flatten(order='F')
    
    print(f"Writing BIN: {bin_path}")
    with open(bin_path, 'wb') as f:
        # Write cols (uint32_t, 4 bytes, little-endian)
        f.write(struct.pack('<I', cols))
        # Write rows (uint32_t, 4 bytes, little-endian)
        f.write(struct.pack('<I', rows))
        # Write raw float array (little-endian)
        f.write(col_major_data.tobytes())
    print("Conversion completed successfully!")

if __name__ == '__main__':
    base_dir = '/home/aditya-pachauri/silent_sentry/src/2_ugv_fleet_brain/bot_navigation/maps'
    tif_path = os.path.join(base_dir, 'synthetic_dem.tif')
    bin_path = os.path.join(base_dir, 'synthetic_dem.bin')
    convert_tif_to_bin(tif_path, bin_path)
