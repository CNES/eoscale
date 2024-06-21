from pathlib import Path

import rasterio


def read_raster(raster_file: Path):
    with rasterio.open(raster_file, "r") as raster_dataset:
        array = raster_dataset.read()
    return array
