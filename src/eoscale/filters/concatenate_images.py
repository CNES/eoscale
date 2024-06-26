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

from pathlib import Path
from typing import List, Union

import numpy as np

from eoscale.core.data_types import VirtualPath
from eoscale.core.eo_executors import n_images_to_m_images_filter
from eoscale.core.manager import EOContextManager

from numpy.typing import DTypeLike


def concatenate_filter(input_buffers: list,
                       input_profiles: list,
                       params: dict) -> List[np.ndarray]:
    """
    Concatenates a list of input buffers along the first axis (bands).

    Parameters
    ----------
    input_buffers : list of np.ndarray
        List of input arrays to be concatenated.
    input_profiles : list
        List of profiles associated with input buffers.
    params : dict
        Dictionary of parameters for the concatenation, including:
            np_type : DTypeLike
                Data type for the output array.

    Returns
    -------
    list of np.ndarray
        List containing the concatenated array.
    """
    res = np.concatenate(input_buffers, axis=0, dtype=params["np_type"]).squeeze()
    return [res]


def concatenate_profile(input_profiles: list,
                        params: dict) -> List[dict]:
    """
    Generates a concatenated profile based on input profiles and parameters.

    Parameters
    ----------
    input_profiles : list
        List of input rasterio profiles.
    params : dict
        Dictionary of parameters for the profile, including:
            np_type : DTypeLike
                Data type for the output profile.

    Returns
    -------
    list[dict]
        A dictionary representing the concatenated profile.
    """
    profile = input_profiles[0]
    profile['dtype'] = params["np_type"]
    profile['count'] = sum(input_profile["count"] for input_profile in input_profiles)
    return [profile]


def concatenate_images(context: EOContextManager, inputs: List[Union[str, VirtualPath]] ,
                       as_type: DTypeLike = np.float32) -> VirtualPath:
    """
    Concatenates a list of input images into a single output image.

    Parameters
    ----------
    context : EOContextManager
        Context manager for handling input and output.
    inputs : list of str or VirtualPath
        List of input image paths.
    as_type : DTypeLike, optional
        Data type for the output image. Default is np.float32.

    Returns
    -------
    VirtualPath
        The path to the output virtual file.

    Raises
    ------
    ValueError
        If more than one output path is generated.

    Example
    -------
    >>> with EOContextManager(nb_workers=4, tile_mode=True) as eoscale_manager:
    >>>     concatenate_vpath = concatenate_images(eoscale_manager, ['/path/to/input.tif',
    >>>                                                              '/path/to/other_input.tif'])
    >>>     concatenate_array = eoscale_manager.get_array(concatenate_vpath)
    """
    imgs = []
    for input_file in inputs:
        if Path.exists(Path(input_file)):
            imgs.append(context.open_raster(raster_path=input_file))
        else:
            imgs.append(input_file)
    v_path = n_images_to_m_images_filter(inputs=imgs,
                                         image_filter=concatenate_filter,
                                         filter_parameters={"np_type": as_type},
                                         generate_output_profiles=concatenate_profile,
                                         context_manager=context,
                                         filter_desc="Concatenate processing...")
    if len(v_path) > 1:
        raise ValueError("concatenate output must be unique")
    return v_path[0]
