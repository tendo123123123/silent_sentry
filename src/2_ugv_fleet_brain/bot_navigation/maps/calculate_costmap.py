import numpy as np
import rasterio
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 calculate_costmap.py <slope_tif> <output_tif>")
        sys.exit(1)
        
    slope_tif = sys.argv[1]
    output_tif = sys.argv[2]
    
    with rasterio.open(slope_tif) as src:
        slope = src.read(1)
        meta = src.meta
        
    # Maps 0 degrees to 255 (White) and 30 degrees to 0 (Black)
    # maximum(0, 255 - ((A/30.0)**2)*255)
    cost = np.maximum(0, 255 - ((slope / 30.0) ** 2) * 255)
    cost = cost.astype(np.uint8)
    
    meta.update(dtype=rasterio.uint8, count=1, nodata=None)
    
    with rasterio.open(output_tif, 'w', **meta) as dst:
        dst.write(cost, 1)

if __name__ == "__main__":
    main()
