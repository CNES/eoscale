from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

def read_raster(raster_file: Path):
    with rasterio.open(raster_file, "r") as raster_dataset:
        array = raster_dataset.read()
    return array


def create_raster(data:np.ndarray, output_file:Path):
    """
    Creates a raster file using the provided numpy array data.

    This function creates a GeoTIFF raster file with the given data, using a predefined
    coordinate reference system (CRS) and spatial transformation.

    Parameters
    ----------
    data : np.ndarray
        Numpy array containing the raster data to be written (bands, height, width)

    output_file : Path
        Path to the output raster file where the data will be saved.

    Raises
    ------
    ValueError
        If the dimensions of `data` do not match the expected raster dimensions.

    Example
    -------
    >>> data = np.random.rand(1, 512, 512)  # Example 512x512 raster data
    >>> output_file = Path("/path/to/output.tif")
    >>> create_raster(data, output_file)
    """
    top_left_x, top_left_y = 10.0, 50.0
    x_res, y_res = 0.01, 0.01
    transform = Affine.translation(top_left_x, top_left_y) * Affine.scale(x_res, -y_res)
    crs = "EPSG:4326"
    with rasterio.open(
            output_file, "w",
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=data.shape[0],
            dtype=data.dtype,
            crs=crs,
            transform=transform
    ) as dst:
        dst.write(data)
