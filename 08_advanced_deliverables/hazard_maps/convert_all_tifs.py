#!/usr/bin/env python3
"""
Batch convert all GeoTIFF hazard maps to JPEG/PNG

Usage:
    python convert_all_tifs.py
"""

import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np

def convert_geotiff_to_images(tif_file, output_dir, quality=150):
    """Convert single GeoTIFF to JPEG and PNG"""

    if not os.path.exists(tif_file):
        print(f"⚠️ File not found: {tif_file}")
        return

    try:
        # Read GeoTIFF
        with rasterio.open(tif_file) as src:
            data = src.read(1)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, cmap='viridis')
        ax.set_title(os.path.basename(tif_file))
        plt.colorbar(im, ax=ax)

        # Save JPEG
        jpg_file = tif_file.replace('.tif', '.jpg')
        plt.savefig(jpg_file, dpi=quality, format='jpg', bbox_inches='tight')
        print(f"✅ {jpg_file}")

        # Save PNG
        png_file = tif_file.replace('.tif', '.png')
        plt.savefig(png_file, dpi=quality, format='png', bbox_inches='tight')
        print(f"✅ {png_file}")

        plt.close()

    except Exception as e:
        print(f"❌ Error processing {tif_file}: {e}")

# Find all TIF files
hazard_dir = "/raid/home/srmist57/Chandrayan-3/LunarSense3_FullPipeline/08_advanced_deliverables/hazard_maps"
tif_files = [
    os.path.join(hazard_dir, f) for f in os.listdir(hazard_dir) 
    if f.endswith('.tif')
]

print(f"Converting {len(tif_files)} GeoTIFF files...
")

for tif_file in tif_files:
    convert_geotiff_to_images(tif_file)

print("
✅ Batch conversion complete!")
