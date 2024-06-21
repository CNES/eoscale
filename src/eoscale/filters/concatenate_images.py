import numpy as np

from eoscale.eo_executors import n_images_to_m_images_filter
from eoscale.manager import EOContextManager

VirtualPath = str
from numpy.typing import DTypeLike
def concatenate_filter(input_buffers: list,
                       input_profiles: list,
                       params: dict) -> list[np.ndarray] :
    res = np.concatenate(input_buffers, axis=0, dtype=params["np_type"]).squeeze()
    return [res]

def concatenate_profile(input_profiles: list,
                        params: dict) -> dict:
    """ """
    profile = input_profiles[0]
    profile['dtype'] = params["np_type"]
    profile['count'] = sum(input_profile["count"] for input_profile in input_profiles)
    return [profile]

def concatenate_images(context: EOContextManager, inputs: list[str] | list[VirtualPath], as_type:DTypeLike=np.float32) -> VirtualPath:
    imgs = [context.open_raster(raster_path=img) for img in inputs]
    v_path = n_images_to_m_images_filter(inputs=imgs,
                                         image_filter=concatenate_filter,
                                         filter_parameters={"np_type":as_type},
                                         generate_output_profiles=concatenate_profile,
                                         context_manager=context,
                                         filter_desc="Concatenate processing...")
    if len(v_path) > 1:
        raise ValueError("concatenate output must be unique")
    return v_path[0]