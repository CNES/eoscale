import numpy
from rasterio.fill import fillnodata

import eoscale.shared as eosh
import eoscale.manager as eom
import eoscale.eo_executors as eoexe


def nodata_filter(input_buffers: list, 
                  input_profiles: list, 
                  params: dict) -> numpy.ndarray :
    """ """
    input_img = input_buffers[0]
    return numpy.where(input_img[0] == input_profiles[0]["nodata"], 1, 0 ).astype(numpy.uint8)

def nodata_profile(input_profiles: list,
                   params: dict) -> dict:
    """ """
    mask_profile = input_profiles[0]
    mask_profile['dtype'] = numpy.uint8
    mask_profile['nodata'] = None
    return mask_profile

###############################################################################################

import copy

def fillnodata_filter(input_buffers: list,
                      input_profiles: list,
                      params: dict) -> numpy.ndarray:
    """ """
    # For in place method, you need to create a deep copy with Python.
    img_with_holes = copy.deepcopy(input_buffers[0][0])
    nodata_mask = input_buffers[1][0]
    return fillnodata(img_with_holes, 
                      mask = numpy.where(nodata_mask > 0, 0, 1).astype(numpy.uint8), 
                      max_search_distance = params['max_search_distance'],
                      smoothing_iterations= params["smoothing_iterations"])

def fillnodata_profile(input_profiles: list,
                       params: dict):
    return input_profiles[0]

###############################################################################################

import scipy
# Definition of the function to map that is simply an uniform filter here
def uniform_filter(input_buffers: list, 
                   input_profiles: dict, 
                   params: dict) -> list:
    """ """
    output_buffer = scipy.ndimage.uniform_filter(input_buffers[0][0], size=params["size"])
    return output_buffer

##############################################################################################

if __name__ == "__main__":

    # Change with your one channel image
    input_image_path: str = "/work/scratch/env/lassalp/bulldozer_workspace/bulldozer_sandbox/debug_ny/dsm_NEW-YORK_tuile_1.tif"
    output_nodata_mask: str = "/work/scratch/env/lassalp/bulldozer_workspace/bulldozer_sandbox/eoscale/nodata_mask.tif"
    output_filled_dsm: str = "/work/scratch/env/lassalp/bulldozer_workspace/bulldozer_sandbox/eoscale/filled_dsm.tif"
    output_image_path: str = "/work/scratch/env/lassalp/bulldozer_workspace/bulldozer_sandbox/eoscale/smooth_dsm.tif"

    nb_workers: int = 8
    tile_mode: bool = True


    #############################################################################################################################
    eoscale_manager = eom.EOContextManager(nb_workers = nb_workers, 
                                           tile_mode = tile_mode)

    eoscale_manager.start()
    #############################################################################################################################

    img_1 = eoscale_manager.open_raster(raster_path = input_image_path)

    ### NoData mask filter

    nodata_outputs = eoexe.n_images_to_m_images_filter(inputs = [img_1],
                                                       image_filter = nodata_filter,
                                                       generate_output_profiles = nodata_profile,
                                                       context_manager = eoscale_manager,
                                                       filter_desc= "Nodata processing...")
    
    # Flush the mask to disk
    eoscale_manager.write(key = nodata_outputs[0], img_path = output_nodata_mask)

    ### Fill no data filter

    fillnodata_parameters: dict = {
        "max_search_distance": 100.0,
        "smoothing_iterations": 0
    }
    fillnodata_margin: int = 100


    fillnodata_outputs = eoexe.n_images_to_m_images_filter(inputs = [img_1, nodata_outputs[0]],
                                                           image_filter = fillnodata_filter,
                                                           filter_parameters = fillnodata_parameters,
                                                           generate_output_profiles = fillnodata_profile,
                                                           context_manager = eoscale_manager,
                                                           stable_margin = fillnodata_margin,
                                                           filter_desc= "Fill nodata processing...")
    
    ### We do not need to have the input image in memory and neither the mask for the next filter
    ### We can therefore ask the context manager to release them
    eoscale_manager.release(key = img_1)
    eoscale_manager.release(key = nodata_outputs[0])

    # Flush the filled dsm to disk
    eoscale_manager.write(key = fillnodata_outputs[0], img_path = output_filled_dsm)

    ### Uniform filter
    uniform_parameters: dict = {
        "size": 3
    }
    uniform_margin : int = 1

    outputs = eoexe.n_images_to_m_images_filter(inputs = [fillnodata_outputs[0]], 
                                                image_filter = uniform_filter,
                                                filter_parameters = uniform_parameters,
                                                stable_margin = uniform_margin,
                                                context_manager = eoscale_manager,
                                                filter_desc= "Uniform filter processing...")
    
    # Flush the smooth filter to disk
    eoscale_manager.write(key = outputs[0], img_path = output_image_path)

    #############################################################################################################################
    # All shared resources are automatically released
    eoscale_manager.end()

