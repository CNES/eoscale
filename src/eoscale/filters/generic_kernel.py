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

from typing import Literal, Callable, Optional, Any, List, Dict, Union
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike
from scipy.ndimage import generic_filter

from eoscale.data_types import VirtualPath
from eoscale.eo_executors import n_images_to_m_images_filter
from eoscale.manager import EOContextManager




def sliding_window_reduce_with_kernel(arr, func: Callable, kernel_size: tuple,
                                      mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"],
                                      cval: int,
                                      func_kwarg: Optional[Dict[str, Any]] = None,
                                      output_dtype: DTypeLike = np.float32) -> np.ndarray:
    """
    Applies a sliding window reduction using a specified kernel and function.

    Parameters
    ----------
    arr : np.ndarray
        Input array to apply the sliding window reduction.
    func : Callable
        Function to apply over the kernel window.
    kernel_size : tuple
        The size of the kernel window.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        The mode parameter determines how the array borders are handled.
    cval : int
        Value to fill past edges of input if mode is 'constant'.
    func_kwarg : dict[str, Any], optional
        Additional keyword arguments to pass to the function.
    output_dtype : DTypeLike
        Data type for the output. Default is np.float32.

    Returns
    -------
    np.ndarray
        The array after applying the sliding window reduction.
    """
    if func_kwarg is None:
        func_kwarg = {}

    result = generic_filter(arr, func, size=kernel_size,
                            mode=mode, cval=cval,
                            extra_keywords=func_kwarg).astype(output_dtype)
    return result


def kernel_filter(input_buffers: list,
                  input_profiles: list,
                  params: dict) -> List[np.ndarray]:
    return [sliding_window_reduce_with_kernel(img, params["func"],
                                              params["kernel_shape"], params["mode"], params["cval"],
                                              params["func_kwarg"]) for img in
            input_buffers]


def generic_profile(input_profiles: list,
                    params: dict) -> dict:
    """ """
    profile = input_profiles[0]
    profile['dtype'] = params["np_type"]
    return [profile] * len(input_profiles)


def generic_kernel_filter(context: EOContextManager,
                          inputs: List[Union[str, VirtualPath]],
                          func: Callable,
                          kernel_radius: int = 1,
                          mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"] = "constant",
                          cval=0.0,
                          dtype: DTypeLike = np.float32,
                          func_kwarg: Optional[Dict[str, Any]] = None
                          ) -> List[VirtualPath]:
    """Applies a sliding window reduction using a specified kernel and function to a list of input.

    Warning
    -------
    Strong hypothesis: all input image are in the same geometry and have the same size

    Parameters
    ----------
    context : EOContextManager
        Context manager for handling input and output.
    inputs : list of str or list of VirtualPath
        List of input image paths.
    func : Callable
        Function to apply over the kernel window.
    kernel_radius : int, optional
        Radius of the kernel window. Default is 1.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled. Default is 'constant'.
            - 'reflect'  (d c b a | a b c d | d c b a)

               The input is extended by reflecting about the edge of the last pixel. This mode is also
               sometimes referred to as half-sample symmetric.

            - 'constant' (k k k k | a b c d | k k k k)

               The input is extended by filling all values beyond the edge with the same constant value,
               defined by the cval parameter.

            - 'nearest'  (a a a a | a b c d | d d d d)

               The input is extended by replicating the last pixel.

            - 'mirror'   (d c b | a b c d | c b a)

               The input is extended by reflecting about the center of the last pixel.
               This mode is also sometimes referred to as whole-sample symmetric.

            - 'wrap'     (a b c d | a b c d | a b c d)

               The input is extended by wrapping around to the opposite edge.

    cval : float, optional
        Value to fill past edges of input if mode is 'constant'. Default is 0.0.
    dtype : DTypeLike, optional
        Data type for the output profile. Default is np.float32.
    func_kwarg : dict[str, Any], optional
        Additional keyword arguments to pass to the function.

    Returns
    -------
    list[VirtualPath]
        The paths to the output virtual files.

    Note
    ----
    under the hood `scipy.ndimage.generic_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter.html>`_
    is called

    Example
    -------
    >>> # sum in a 5x5 sliding window
    >>> with EOContextManager(nb_workers=4, tile_mode=True) as eoscale_manager:
    >>>     out_vpath = generic_kernel_filter(eoscale_manager,
    >>>                                       ['/path/to/input.tif'],
    >>>                                       np.sum, kernel_radius=2)[0]
    >>>     arr_margin = eoscale_manager.get_array(out_vpath)
    """
    imgs = []
    for input_file in inputs:
        if Path.exists(Path(input_file)):
            imgs.append(context.open_raster(raster_path=input_file))
        else:
            imgs.append(input_file)
    kernel_shape = (1, 1 + 2 * kernel_radius, 1 + 2 * kernel_radius)
    return n_images_to_m_images_filter(inputs=imgs,
                                       image_filter=kernel_filter,
                                       filter_parameters={"func": func,
                                                          "kernel_shape": kernel_shape,
                                                          "mode": mode,
                                                          "cval": cval,
                                                          "np_type": dtype,
                                                          "func_kwarg": func_kwarg},
                                       generate_output_profiles=generic_profile,
                                       context_manager=context,
                                       stable_margin=kernel_radius,
                                       filter_desc="Generic processing...")
