import cv2
import numpy as np
import sys

print("Loading 32-bit EXR...")
# IMREAD_UNCHANGED forces OpenCV to respect the 32-bit float depth
img = cv2.imread('synthetic_dem.exr', cv2.IMREAD_UNCHANGED)

if img is None:
    print("Error: Could not read synthetic_dem.exr")
    sys.exit(1)

# Blender exports EXR as RGB even if it is a black and white emission. 
# We isolate a single channel to ensure the final TIFF is a 1-band DEM.
if len(img.shape) == 3:
    img = img[:, :, 0]

print("Writing 32-bit GeoTIFF...")
# OpenCV automatically writes a 32-bit Float TIFF when handed a float32 NumPy array
success = cv2.imwrite('synthetic_dem.tif', img)

if success:
    print("Translation Successful. synthetic_dem.tif is ready.")
else:
    print("Error writing TIFF.")
