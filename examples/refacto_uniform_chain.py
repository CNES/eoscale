import eoscale.manager as eom
import eoscale.eo_executors as eoexe

import scipy
# Definition of the function to map that is simply an uniform filter here
def uniform_filter(input_buffers: list, params: dict) -> list:
    output_buffer = scipy.ndimage.uniform_filter(input_buffers[0][0], size=params["size"])
    output_buffer = output_buffer.reshape((1, output_buffer.shape[0], output_buffer.shape[1]))
    return [output_buffer]

if __name__ == "__main__":

    # Change with your one channel image
    # input_image_path: str = "/work/scratch/env/lassalp/bulldozer_workspace/bulldozer_sandbox/debug_ny/dsm_NEW-YORK_tuile_1.tif"
    # output_image_path: str = "/work/scratch/env/lassalp/bulldozer_workspace/bulldozer_sandbox/eoscale/smooth_dsm.tif"
    input_image_path: str = "/work/scratch/lassalp/AI4GEO_WORKSPACE/input/dsm.tif"
    output_image_path: str = "/work/scratch/lassalp/AI4GEO_WORKSPACE/output/smooth_dsm.tif"

    nb_workers: int = 8
    tile_mode: bool = True


    #############################################################################################################################
    eoscale_manager = eom.EOContextManager(nb_workers = nb_workers, 
                                           tile_mode = tile_mode)

    eoscale_manager.start()
    #############################################################################################################################

    img_1 = eoscale_manager.open_raster(raster_path = input_image_path)

    # First filter call

    filter_1_parameters: dict = {
        "size": 3
    }
    stable_margin_1 : int = 1

    outputs = eoexe.n_images_to_m_images_filter(inputs = [img_1], 
                                                image_filter = uniform_filter,
                                                filter_parameters = filter_1_parameters,
                                                generate_output_profiles = None,
                                                concatenate_filter = None,
                                                stable_margin = stable_margin_1,
                                                context_manager = eoscale_manager)


    #############################################################################################################################
    # All shared resources are automatically released
    eoscale_manager.end()

