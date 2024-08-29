#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of EOSCale
# (see https://github.com/CNES/eoscale).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import warnings
from typing import Optional

import rasterio
from rasterio.profiles import DefaultGTiffProfile
from rasterio.transform import Affine
import numpy
from collections import namedtuple

MpTile = namedtuple('MpTile', ["start_x", "start_y", "end_x", "end_y", "top_margin", "right_margin", "left_margin",
                               "bottom_margin"])

JSON_NONE: str = "none"


def deprecated(message: Optional[str] = None):
    """
    Decorator to mark functions as deprecated.

    This decorator issues a warning when the decorated function is called, indicating
    that the function is deprecated and may be removed in a future version.

    Parameters
    ----------
    message : str, optional
        An optional message to include in the warning, providing additional information
        about the deprecation, such as suggested alternatives.

    Returns
    -------
    function
        The decorated function which will issue a deprecation warning when called.

    Examples
    --------
    >>> @deprecated("Use `new_function` instead.")
    ... def old_function():
    ...     pass
    >>> old_function()
    __main__:2: DeprecationWarning: Call to deprecated function old_function: Use `new_function` instead.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            warnings.warn(
                f"Call to deprecated function {func.__name__} {f': {message}' if message else ''}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)

        return wrapped_func

    return decorator


def rasterio_profile_to_dict(profile: rasterio.DatasetReader.profile) -> dict:
    """
        Convert a rasterio profile to a serializable python dictionnary
        needed for storing in a chunk of memory that will be shared among
        processes
    """
    metadata = dict()
    for key, value in profile.items():
        if key == "crs":
            if value is None:
                metadata['crs'] = None
            else:
                # call to to_authority() gives ('EPSG', '32654')
                metadata['crs'] = int(profile['crs'].to_authority()[1])
        elif key == "transform":
            if value is None:
                metadata['transform'] = None
            else:
                metadata['transform_1'] = profile['transform'][0]
                metadata['transform_2'] = profile['transform'][1]
                metadata['transform_3'] = profile['transform'][2]
                metadata['transform_4'] = profile['transform'][3]
                metadata['transform_5'] = profile['transform'][4]
                metadata['transform_6'] = profile['transform'][5]
        elif key == "nodata":
            if value is None:
                metadata[key] = JSON_NONE
            else:
                metadata[key] = value
        elif key == "dtype":
            if not isinstance(value, str):
                metadata[key] = numpy.dtype(value).name
            else:
                metadata[key] = value
        else:
            metadata[key] = value
    return metadata


def dict_to_rasterio_profile(metadata: dict) -> rasterio.DatasetReader.profile:
    """
        Convert a serializable dictionnary to a rasterio profile
    """
    rasterio_profile = {}
    for key, value in metadata.items():
        if key == "crs":
            if value != None:
                rasterio_profile["crs"] = rasterio.crs.CRS.from_epsg(metadata['crs'])
            else:
                rasterio_profile["crs"] = None
        elif key == "transform_1":
            rasterio_profile['transform'] = rasterio.Affine(metadata['transform_1'],
                                                            metadata['transform_2'],
                                                            metadata['transform_3'],
                                                            metadata['transform_4'],
                                                            metadata['transform_5'],
                                                            metadata['transform_6'])
        elif key.startswith("transform_"):
            continue
        elif key == "transform":
            if value is None:
                rasterio_profile["transform"] = None
        elif key == "nodata":
            if value == JSON_NONE:
                rasterio_profile[key] = None
            else:
                rasterio_profile[key] = value
        else:
            rasterio_profile[key] = value

    return rasterio_profile


def create_default_rasterio_profile(nb_bands: int,
                                    dtype: numpy.dtype,
                                    xstart: float,
                                    ystart: float,
                                    xsize: int,
                                    ysize: int,
                                    resolution: float,
                                    nodata: float = None) -> rasterio.DatasetReader.profile:
    transform = Affine.translation(xstart, ystart)
    transform = transform * Affine.scale(resolution, -resolution)
    profile = DefaultGTiffProfile(
        count=nb_bands,
        dtype=dtype,
        width=xsize,
        height=ysize,
        transform=transform,
        nodata=nodata
    )

    return profile
