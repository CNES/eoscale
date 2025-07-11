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

import numpy
from rasterio.fill import fillnodata

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

def stats_filter(input_buffers : list, 
                 input_profiles: list, 
               filter_parameters: dict):

    """ """
    img = input_buffers[0]
    min_value : float = numpy.min(img)
    max_value : float = numpy.max(img)

    return [min_value, max_value]

##############################################################################################

def stats_concatenate(output_scalars, chunk_output_scalars, tile):
    output_scalars[0] = min( output_scalars[0], chunk_output_scalars[0] )
    output_scalars[1] = max( output_scalars[1], chunk_output_scalars[1] )

##############################################################################################

if __name__ == "__main__":

    # Change with your one channel image
    input_image_path: str = "./data/dsm.tif"
    output_nodata_mask: str = "./data/outputs/nodata_mask.tif"
    output_filled_dsm: str = "./data/outputs/filled_dsm.tif"
    output_image_path: str = "./data/outputs/smooth_dsm.tif"

    nb_workers: int = 8
    tile_mode: bool = True


    #############################################################################################################################

    with eom.EOContextManager(nb_workers = nb_workers, tile_mode = tile_mode) as eoscale_manager:


        img_1 = eoscale_manager.open_raster(raster_path = input_image_path)

        ### Step 1 : NoData filter

        nodata_outputs = eoexe.n_images_to_m_images_filter(inputs = [img_1],
                                                           image_filter = nodata_filter,
                                                           generate_output_profiles = nodata_profile,
                                                           context_manager = eoscale_manager,
                                                           filter_desc= "Nodata processing...")
        
        # Flush the mask to disk
        eoscale_manager.write(key = nodata_outputs[0], img_path = output_nodata_mask)

        ### Step 2: Fill no data filter

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

        ### Step 3: Uniform filter
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

        eoscale_manager.release(key = fillnodata_outputs[0])

        # Step 4: Stats computation

        stats = eoexe.n_images_to_m_scalars(inputs = [outputs[0]],
                                            image_filter = stats_filter,
                                            nb_output_scalars = 2,
                                            concatenate_filter = stats_concatenate,
                                            context_manager = eoscale_manager,
                                            filter_desc= "Min value processing...")

        print(stats[0], stats[1])

    #############################################################################################################################
    # All shared resources are automatically released

