from typing import Callable
import concurrent.futures
import tqdm
import numpy

import eoscale.shared as eosh
import eoscale.eo_io as eoio
import eoscale.utils as eotools


def compute_mp_strips(input_vpath: str,
                      stable_margin: int,
                      nb_workers: int):
    """
        Given an input eoscale virtual path an nb_workers,
        this method computes the list of strips that will
        be processed in parallel within a stream strip or tile 
    """

    eo_shared_inst = eosh.EOShared(eoshared_vpath=input_vpath)
    metadata = eo_shared_inst.get_metadata()

    image_width = int(metadata['width'])
    image_height = int(metadata['height'])
    strip_height = image_height // nb_workers

    strips = []
    start_y: int = None
    end_y: int = None
    top_margin: int = None
    bottom_margin: int = None

    for worker in range(nb_workers):

        start_y = worker * strip_height

        if worker == nb_workers - 1:
            end_y = image_height - 1            
        else:
            end_y = (worker + 1) * strip_height - 1
        
        top_margin = stable_margin if start_y - stable_margin >= 0 else start_y
        bottom_margin = stable_margin if end_y + stable_margin <= image_height - 1 else image_height - 1 - end_y


        strips.append(eotools.MpStrip(start_x = 0, 
                              start_y = start_y, 
                              end_x = image_width - 1, 
                              end_y = end_y, 
                              top_margin = top_margin, 
                              bottom_margin = bottom_margin))
                    
    # Always close the shared resource
    eo_shared_inst.close()

    return strips

def copy_input_metadatas(input_vpaths: list) -> list:
    """ Naive copy of input metadatas into a list """
    eo_shared_instances = [ eosh.EOShared(input_vpath) for input_vpath in input_vpaths ]
    output_mtds = [ eo_shared_instances[i].get_metadata() for i in range(len(input_vpaths)) ]

    for eo_shared_inst in eo_shared_instances:
        eo_shared_inst.close()
    
    return output_mtds

def allocate_outputs(metadatas: list) -> list:
    """
        Given a list of transformed metadatas, this function
        returns a list of allocated output numpy arrays 
    """
    outputs = []
    for mtd in metadatas:
        outputs.append( numpy.zeros(shape = (mtd["count"], mtd["height"], mtd["width"]),
                                    dtype = mtd["dtype"]) )

    return outputs

def default_reduce(outputs: list, 
                   chunk_output_buffers: list, 
                   strip: eotools.MpStrip) -> None:
    """ Fill the outputs buffer with the results provided by the map filter from a strip """
    for c in range(len(chunk_output_buffers)):
        outputs[c][:, strip.start_y: strip.end_y + 1,:] = chunk_output_buffers[c][:,:,:]

def execute_map_filter_n_images(map_filter: Callable,
                                map_params: dict,
                                input_vpaths: list,
                                s: eotools.MpStrip) -> list :

    """ This method maps the filter on the input virtual paths and return a list of output buffers """
    eo_shared_instances = [ eosh.EOShared(input_vpath) for input_vpath in input_vpaths ]

    input_buffers = [ eo_shared_instances[i].get_array(strip=s) for i in range(len(input_vpaths)) ]

    output_buffers = map_filter( input_buffers, map_params )

    for eo_shared_inst in eo_shared_instances:
        eo_shared_inst.close()
    
    # Extract the stable area for each output buffer
    stable_start_y: int = None
    stable_end_y: int = None
    for i in range(len(output_buffers)):
        # Extract the stable area
        stable_start_y = s.top_margin
        stable_end_y = stable_start_y + s.end_y - s.start_y + 1
        output_buffers[i] = output_buffers[i][:, stable_start_y:stable_end_y, :]
    
    return output_buffers, s

def n_images_to_m_images_filter(input_vpaths: list = None,
                                map_filter: Callable = None,
                                map_params: dict = None,
                                metatada_transformer: Callable = None,
                                reduce_filter: Callable = None,
                                stable_margin: int = None,
                                nb_workers: int = None,
                                map_desc: str = "Map multi processing...") -> list:
    """
        Generic paradigm to process n images providing m resulting images using a paradigm
        similar to the good old map/reduce

        map filter is processed in parallel
        reduce filter is processed by the master node to aggregate results

        Strong hypothesis: all input image are in the same geometry and have the same size
    """
    if len(input_vpaths) < 1:
        raise ValueError("You must give at least one eoscale virtual path")

    # compute the strips
    strips = compute_mp_strips(input_vpath = input_vpaths[0],
                               stable_margin = stable_margin,
                               nb_workers = nb_workers)
    
    # call the metadata transformer if provided, 
    # if not then input metadata are the same of output metadata
    output_metadatas = None
    if metatada_transformer is not None:
        output_metadatas = metatada_transformer(input_vpaths, map_params)
    else:
        output_metadatas = copy_input_metadatas(input_vpaths)

    # Allocate in memory for the master process the outputs
    outputs = allocate_outputs(output_metadatas)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=nb_workers) as executor:

        futures = { executor.submit(execute_map_filter_n_images,
                                    map_filter,
                                    map_params,
                                    input_vpaths, 
                                    s) for s in strips }

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=map_desc):

            chunk_output_buffers, strip = future.result()

            if reduce_filter is not None:
                reduce_filter( outputs, chunk_output_buffers, strip )
            else:
                default_reduce(outputs, chunk_output_buffers, strip )
    
    # Create eo_shared instances of the outputs and return it to the user
    output_eoshared_instances = []
    for o in range(len(outputs)):
        output_eoshared_instances.append(eosh.EOShared())
        output_eoshared_instances[-1].create_from_in_memory_array(big_array = outputs[o], user_metadata=output_metadatas[o])
        outputs[o] = None
    
    return output_eoshared_instances


    

        