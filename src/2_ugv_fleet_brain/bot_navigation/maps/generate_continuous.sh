#!/bin/bash

# Configuration
INPUT_TIF="synthetic_dem.tif"
SLOPE_TIF="slope_map.tif"
CONTINUOUS_TIF="continuous_costmap.tif"
OUTPUT_PGM="continuous_planner_map.pgm"

# 1. Verify Input
if [ ! -f "$INPUT_TIF" ]; then
    echo "Error: $INPUT_TIF not found."
    exit 1
fi

# 2. Generate Slope Map
# Calculates degrees (0-90) for each pixel
gdaldem slope "$INPUT_TIF" "$SLOPE_TIF" -of GTiff -s 1.0 -compute_edges

# 3. Generate Continuous Costmap (Quadratic Inverted)
# Maps 0 degrees to 255 (White) and 30 degrees to 0 (Black)
python3 calculate_costmap.py "$SLOPE_TIF" "$CONTINUOUS_TIF"

# 4. Convert to PGM
gdal_translate -of PNM "$CONTINUOUS_TIF" "$OUTPUT_PGM"

# Cleanup intermediate TIFs if only the PGM is required
# rm "$SLOPE_TIF" "$CONTINUOUS_TIF"

echo "Continuous costmap generation complete: $OUTPUT_PGM"