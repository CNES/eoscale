import eoscale.manager as eom
import eoscale.eo_executors as eoexe

import scipy
import numpy as np

# Definition of the function to map that is simply an uniform filter here
def uniform_filter(input_buffers: list, 
                   input_profiles: dict, 
                   params: dict) -> list:
    """ """
    input_buffers[0][0] = scipy.ndimage.uniform_filter(input_buffers[0][0], size=params["size"])
    input_profiles[0]["nodata"] = "-40000.0"

if __name__ == "__main__":

    # Change with your one channel image
    input_image_path: str = "/work/scratch/env/lassalp/bulldozer_workspace/bulldozer_sandbox/eoscale/smooth_dsm.tif"

    nb_workers: int = 8
    tile_mode: bool = True

    with eom.EOContextManager(nb_workers = nb_workers, tile_mode = tile_mode) as eoscale_manager:

        img_1 = eoscale_manager.open_raster(raster_path = input_image_path)

        uniform_parameters: dict = {
            "size": 3
        }

        # Run a sequential in place filter
        [img_1] = eoexe.in_place_sequential_image_filter(inputs = [img_1],
                                                         image_filter = uniform_filter,
                                                         filter_parameters = uniform_parameters,
                                                         context_manager = eoscale_manager,
                                                         filter_desc= "Sequential inplace uniform processing...")