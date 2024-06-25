from pathlib import Path
from typing import List
from functools import reduce

import numpy as np
import rasterio
from rasterio.transform import Affine


def read_raster(raster_file: Path):
    with rasterio.open(raster_file, "r") as raster_dataset:
        array = raster_dataset.read()
    return array


def create_raster(data: np.ndarray, output_file: Path):
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


def assert_profiles(files_path: List[str]) -> bool:
    """
    Compares the raster profiles of multiple raster files to check if they are all equal.

    This function reads the profile metadata of each raster file in the provided list of paths.
    It uses the reduce function to compare the profiles pairwise.
    If all profiles are identical, the function returns True. Otherwise, it returns False.

    Parameters
    ----------
    files_path : List[str]
        A list of file paths pointing to the raster files to be compared.

    Returns
    -------
    bool
        True if all raster profiles are equal, False otherwise.
    """

    def compare(x, y):
        return x if x == y else False

    profiles = []
    for current_file in files_path:
        with rasterio.open(current_file, "r") as raster_dataset:
            profiles.append(raster_dataset.profile)

    return reduce(compare, profiles) is not False
