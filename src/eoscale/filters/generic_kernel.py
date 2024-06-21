from typing import Literal, Callable, Optional, Any

import numpy as np

from eoscale.eo_executors import n_images_to_m_images_filter
from eoscale.manager import EOContextManager
from numpy.typing import DTypeLike
from scipy.ndimage import generic_filter

VirtualPath = str


def sliding_window_reduce_with_kernel(arr, func: Callable, kernel_size: tuple,
                                      mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"],
                                      cval: int, func_kwarg: Optional[dict[str, Any]] = None):
    """
    """
    if func_kwarg is None:
        func_kwarg = {}

    result = generic_filter(arr, func, size=kernel_size, mode=mode, cval=cval, extra_keywords=func_kwarg)
    return result


def kernel_filter(input_buffers: list,
                  input_profiles: list,
                  params: dict) -> list[np.ndarray]:
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


def generic_kernel_filter(context: EOContextManager, inputs: list[str] | list[VirtualPath],
                          func: Callable,
                          kernel_radius: int = 1,
                          mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"] = "constant",
                          cval=0.0,
                          dtype: DTypeLike = np.float32,
                          func_kwarg: Optional[dict[str, Any]] = None
                          ) -> VirtualPath:
    imgs = [context.open_raster(raster_path=img) for img in inputs]
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
