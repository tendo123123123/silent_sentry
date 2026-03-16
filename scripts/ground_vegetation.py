#!/usr/bin/env python3
"""
ground_vegetation.py  –  Perfectly ground every vegetation model onto the DEM terrain.

Reads:
    src/bot_description/worlds/dem/output_be_sq.tif   (128×128 float32, 0–51 m)
    src/bot_description/worlds/thar_desert.sdf        (1790 vegetation <include> blocks)

Writes:
    src/bot_description/worlds/thar_desert_perfect_ground.sdf

=== Coordinate-to-pixel math ===

Gazebo centres the heightmap at the world origin.  With <size> = (SX, SY, SZ)
and <pos> = (0, 0, 0), the heightmap spans:

    X ∈ [-SX/2,  +SX/2]      →   [-867.65,  +867.65]   (1735.3 m)
    Y ∈ [-SY/2,  +SY/2]      →   [-2071.8,  +2071.8]   (4143.6 m)

The DEM image has shape (nrows, ncols) = (128, 128).

Pixel (0, 0) is the TOP-LEFT corner = (X_min, Y_max) = (-SX/2, +SY/2).
Pixel (nrows-1, ncols-1) is the BOTTOM-RIGHT = (+SX/2, -SY/2).

So for a world position (wx, wy):

    col_frac = (wx - (-SX/2)) / SX  * (ncols - 1)   =  (wx + SX/2) / SX * (ncols-1)
    row_frac = ((-wy) - (-SY/2)) / SY * (nrows - 1)  =  (SY/2 - wy) / SY * (nrows-1)

We bilinearly interpolate the four surrounding pixels to get the ground
elevation at sub-pixel resolution.

The DEM pixel values are already in physical metres (0.0–51.0 m).  With
<size z>=51.0 and <pos z>=0, Gazebo maps:

    z_world  =  pos_z  +  (pixel_value / max_pixel_value) * size_z

Since the .tif is float32 and its actual data range is [0.0, 51.0], and Gazebo
Harmonic uses the raw float value directly (not normalized by max), the world
Z equals the pixel value:

    z_world  =  pixel_value

(Confirmed: the DEM min is 0.0 and max is 51.0, matching <size z>=51.0.)

=== Usage ===

    cd /home/sailesh/silent_sentry
    python3 scripts/ground_vegetation.py

Optionally install to the runtime:

    cp src/bot_description/worlds/thar_desert_perfect_ground.sdf \
       src/bot_description/worlds/thar_desert.sdf
    cp src/bot_description/worlds/thar_desert.sdf \
       install/bot_description/share/bot_description/worlds/thar_desert.sdf
"""

import re
import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Resolve paths relative to this script (expects repo root = scripts/..)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)

DEM_PATH = os.path.join(REPO_ROOT,
    "src", "bot_description", "worlds", "dem", "output_be_sq.tif")
SDF_PATH = os.path.join(REPO_ROOT,
    "src", "bot_description", "worlds", "thar_desert.sdf")
OUT_PATH = os.path.join(REPO_ROOT,
    "src", "bot_description", "worlds", "thar_desert_perfect_ground.sdf")

# ---------------------------------------------------------------------------
# Heightmap physical size (from the SDF <size> tag — MUST match the world)
# ---------------------------------------------------------------------------
HM_SIZE_X = 1735.3   # metres, east–west
HM_SIZE_Y = 4143.6   # metres, north–south
HM_SIZE_Z =   51.0   # metres, vertical span

# ---------------------------------------------------------------------------
# Load DEM
# ---------------------------------------------------------------------------
# Try rasterio first; fall back to PIL if rasterio is unavailable.
try:
    sys.path.insert(0, os.path.join(REPO_ROOT, ".venv", "lib",
                                     "python3.12", "site-packages"))
    import rasterio
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float64)   # shape (nrows, ncols)
    print(f"[INFO] Loaded DEM via rasterio: {dem.shape}, "
          f"dtype={dem.dtype}, range=[{dem.min():.3f}, {dem.max():.3f}]")
except ImportError:
    from PIL import Image
    img = Image.open(DEM_PATH)
    dem = np.array(img, dtype=np.float64)
    # If 8-bit or 16-bit, normalize to [0, HM_SIZE_Z]
    if dem.max() <= 255:
        dem = dem / 255.0 * HM_SIZE_Z
    elif dem.max() <= 65535:
        dem = dem / 65535.0 * HM_SIZE_Z
    print(f"[INFO] Loaded DEM via PIL: {dem.shape}, "
          f"range=[{dem.min():.3f}, {dem.max():.3f}]")

NROWS, NCOLS = dem.shape
print(f"[INFO] DEM grid: {NROWS} rows × {NCOLS} cols")
print(f"[INFO] Heightmap physical size: {HM_SIZE_X} × {HM_SIZE_Y} × {HM_SIZE_Z} m")


# ---------------------------------------------------------------------------
# Bilinear terrain sampler
# ---------------------------------------------------------------------------
def terrain_z(wx: float, wy: float) -> float:
    """
    Given a world-frame position (wx, wy), return the DEM ground height.

    Coordinate mapping (Gazebo heightmap centred at origin):
        col_frac  =  (wx + SX/2) / SX  *  (ncols - 1)
        row_frac  =  (SY/2 - wy) / SY  *  (nrows - 1)

    Bilinear interpolation over the four surrounding pixels ensures
    sub-pixel accuracy.
    """
    # --- World → fractional pixel ------------------------------------------
    col_f = (wx + HM_SIZE_X / 2.0) / HM_SIZE_X * (NCOLS - 1)
    row_f = (HM_SIZE_Y / 2.0 - wy) / HM_SIZE_Y * (NROWS - 1)

    # Clamp to valid pixel range
    col_f = np.clip(col_f, 0.0, NCOLS - 1)
    row_f = np.clip(row_f, 0.0, NROWS - 1)

    # --- Integer corners ---------------------------------------------------
    c0 = int(np.floor(col_f));  c1 = min(c0 + 1, NCOLS - 1)
    r0 = int(np.floor(row_f));  r1 = min(r0 + 1, NROWS - 1)

    # --- Fractional part ---------------------------------------------------
    fc = col_f - c0
    fr = row_f - r0

    # --- Bilinear interpolation --------------------------------------------
    z = (dem[r0, c0] * (1 - fc) * (1 - fr) +
         dem[r0, c1] *      fc  * (1 - fr) +
         dem[r1, c0] * (1 - fc) *      fr  +
         dem[r1, c1] *      fc  *      fr)

    return float(z)


# ---------------------------------------------------------------------------
# Load and process the SDF
# ---------------------------------------------------------------------------
with open(SDF_PATH, "r") as f:
    sdf_content = f.read()

# Match every <include> block that contains a vegetation name (veg_NNNN)
INCLUDE_RE = re.compile(r"(<include>.*?</include>)", re.DOTALL)
NAME_RE    = re.compile(r"<name>veg_\d+</name>")
POSE_RE    = re.compile(
    r"<pose>"
    r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+"   # X
    r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+"   # Y
    r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+"   # Z (to be replaced)
    r"(\S+)\s+"                             # Roll
    r"(\S+)\s+"                             # Pitch
    r"(\S+)"                                # Yaw
    r"</pose>"
)

fixed_count  = 0
total_veg    = 0
max_abs_diff = 0.0
sum_abs_diff = 0.0


def fix_include(match: re.Match) -> str:
    """Replace the Z value in a vegetation <include> block."""
    global fixed_count, total_veg, max_abs_diff, sum_abs_diff

    block = match.group(1)

    # Only process vegetation includes (identified by <name>veg_NNNN</name>)
    if not NAME_RE.search(block):
        return block

    total_veg += 1

    def replace_pose(pm: re.Match) -> str:
        global fixed_count, max_abs_diff, sum_abs_diff

        x_str, y_str, z_str, roll, pitch, yaw = pm.groups()
        wx, wy, old_z = float(x_str), float(y_str), float(z_str)

        # Sample the true ground height from the DEM
        new_z = terrain_z(wx, wy)

        diff = abs(new_z - old_z)
        max_abs_diff = max(max_abs_diff, diff)
        sum_abs_diff += diff
        fixed_count += 1

        return f"<pose>{x_str} {y_str} {new_z:.4f} {roll} {pitch} {yaw}</pose>"

    return POSE_RE.sub(replace_pose, block)


new_sdf = INCLUDE_RE.sub(fix_include, sdf_content)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
with open(OUT_PATH, "w") as f:
    f.write(new_sdf)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
avg_diff = sum_abs_diff / fixed_count if fixed_count else 0.0
print(f"\n{'='*60}")
print(f"  Vegetation models found:  {total_veg}")
print(f"  Poses corrected:          {fixed_count}")
print(f"  Max |ΔZ|:                 {max_abs_diff:.4f} m")
print(f"  Mean |ΔZ|:                {avg_diff:.4f} m")
print(f"  Output written to:        {OUT_PATH}")
print(f"{'='*60}")
print(f"\nTo apply:  cp {OUT_PATH} {SDF_PATH}")
