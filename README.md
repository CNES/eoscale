# EOSCALE

## What is the purpose ?

A situation we have come accross very frequently as remote sensing engineer at CNES is the need to keep one or multiple NumPy arrays in memory for processing large satellite images in a parallel environnement that can be distributed or not.

Because Python multithreading is not simultaneous and not designed for maximising the use of CPUs ([I recommend this nice youtube video](https://www.youtube.com/watch?v=AZnGRKFUU0c)), we choose Python multiprocessing module for scaling our algorithms.

However, having to use multiple processes means we have some limitations when it comes to sharing those NumPy arrays. Fortunately since Python 3.8+, the concept of shared_memory has been introduced to share data structures between processes. It relies on mmap under the hood.

EOScale relies on this concept to propose a class named EOShared to store and share large satellite images and their metadatas between processes.

In addition, EOScale provides a Map/Reduce paradigm that uses a tiling strategy with the concept of stability margin to parallelize local dependency algorithms while ensuring identical results. All the complexity is done for you, you just have to define your algorithm as a callable function that takes as input a list of numpy arrays and a Python dictionnary and that is all !

## Example with documentation

To see how it is easy, just look at this example below that chain 3 uniform filter of different sizes in memory and in parallel with EOScale:

```
import eoscale.shared as eosh
import eoscale.eo_io as eoio
import eoscale.utils as eotools
import eoscale.eo_executors as eoexe

import scipy

# Definition of the function to map that is simply an uniform filter here
def uniform_filter(input_buffers: list, params: dict) -> list:
    output_buffer = scipy.ndimage.uniform_filter(input_buffers[0][0], size=params["size"])
    output_buffer = output_buffer.reshape((1, output_buffer.shape[0], output_buffer.shape[1]))
    return [output_buffer]


if __name__ == "__main__":

    # Change with your one channel image
    input_image_path: str = "/work/scratch/lassalp/AI4GEO_WORKSPACE/input/dsm.tif"
    output_image_path: str = "/work/scratch/lassalp/AI4GEO_WORKSPACE/input/smooth_dsm.tif"

    # Number of cpus to use
    nb_workers: int = 16

    # Map algorithm parameters for the three uniform filters that will be chained in memory    
    map_params_1 = { "size": 3 }
    stable_margin_1: int = 1
    map_params_2 = { "size": 5 }
    stable_margin_2: int = 2
    map_params_3 = { "size" : 7 }
    stable_margin_3: int = 3

    
    # Creation of an instance EOShared from an image path, it will creates 2 shared memory
    # blocs containing the array and the metadata of the image
    # Those shared memory blocs variable will live inside the "with" bloc and will be released from
    # memory automatically. 
    with eosh.ShOpen(resource_path = input_image_path) as dsm_vpath:

        # n_images_to_m_images_filter is a map/reduce filter that takes as input:
        #   - input_vpaths: a list of virtual paths pointing to the shared memory blocs of the input images to process. Thoses images
        #     are in the same referential, and have exactly the same size. [MANDATORY]
        #   - map_filter: a callable filter given by the user (as the uniform_filter declared before) that takes a list of numpy_arrays and
        #     python dictionnary that contains the specific parameters of the filter. In our example, the size of the uniform filter
        #     [MANDATORY]
        #   - map_params: the dictionnary containing the specific parameter of the filter. [MANDATORY if the filter needs to be configured]
        #   - metatada_transformer: A callable function to indicate the number and how the output metadatas are transformed. [OPTIONAL], by default the
        #     input metadatas are copied. The function takes as input the list of input metadatas and a the specific parameters of the
        #     filter.
        #   - reduce_filter: A callable function to describe how the outputs must be aggregated. [OPTIONAL]. This function takes as inputs:
        #           -- A list of the final outputs (numpy arrays)
        #           -- A list of the local outputs to aggregate (numpy arrays)
        #           -- A strip containing the informations: start_x, start_y, end_x, end_y, top_margin, bottom_margin
        #   - stable_margin: an integer value representing the number of pixels to consider around each strip to ensure identical results for each pixel in the strip
        #   - nb_workers: the number of cpus to use for multiprocessing.

        output_shared_instances_1 = eoexe.n_images_to_m_images_filter(input_vpaths = [dsm_vpath], 
                                                                      map_filter = uniform_filter,
                                                                      map_params = map_params_1,
                                                                      metatada_transformer = None,
                                                                      reduce_filter = None,
                                                                      stable_margin = stable_margin_1, 
                                                                      nb_workers = nb_workers)

    # Here, the shared memory blocs pointing by dsm_vpath do not exist anymore in memory.
    # However the resulting eoshared instances return by the first map/reduce (n_images_to_m_images_filter)
    # are still living and will be used for the second map/reduce step. 
    # The chaining is done in memory !

    # From the resulting eoshared instances from the first step, we retrieve the virtual paths
    input_vpaths_2 = [ eor.get_vpath() for eor in output_shared_instances_1 ]

    output_shared_instances_2 = eoexe.n_images_to_m_images_filter(input_vpaths = input_vpaths_2, 
                                                                  map_filter = uniform_filter,
                                                                  map_params = map_params_2,
                                                                  metatada_transformer = None,
                                                                  reduce_filter = None,
                                                                  stable_margin = stable_margin_2, 
                                                                  nb_workers = nb_workers)

    # At this step we do not need the eoshared instances return by the first map/reduce anymore
    # therefore we can release them from memory.
    eotools.release_all(eo_shrd_instances = output_shared_instances_1)

    # A last step for fun
    input_vpaths_3 = [ eor.get_vpath() for eor in output_shared_instances_2 ]
    output_shared_instances_3 = eoexe.n_images_to_m_images_filter(input_vpaths = input_vpaths_3, 
                                                                  map_filter = uniform_filter,
                                                                  map_params = map_params_3,
                                                                  metatada_transformer = None,
                                                                  reduce_filter = None,
                                                                  stable_margin = stable_margin_3, 
                                                                  nb_workers = nb_workers)
                                                            
    eotools.release_all(eo_shrd_instances = output_shared_instances_2)

    # Now we can save to disk the final outputs by giving a list of output paths
    # and a list of eoshared instances to write.
    eoio.write_images(output_paths=[output_image_path], 
                      eo_shared_instances = output_shared_instances_3)


    # Do not forget to release the final eoshared instances.
    eotools.release_all(eo_shrd_instances = output_shared_instances_3)

```





