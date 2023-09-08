import numpy

import eoscale.manager as eom
import eoscale.eo_executors as eoexe


def stats_filter(input_buffers : list, 
                 input_profiles: list, 
               filter_parameters: dict):

    """ """
    img = input_buffers[0]
    min_value : float = numpy.min(img)
    max_value : float = numpy.max(img)

    return [min_value, max_value]

def stats_concatenate(output_scalars, chunk_output_scalars, tile):
    output_scalars[0] = min( output_scalars[0], chunk_output_scalars[0] )
    output_scalars[1] = max( output_scalars[1], chunk_output_scalars[1] )

if __name__ == "__main__":

    input_img_path: str = "/work/scratch/env/lassalp/bulldozer_workspace/bulldozer_sandbox/eoscale/smooth_dsm.tif"

    nb_workers: int = 8
    tile_mode: bool = True

    #############################################################################################################################
    with eom.EOContextManager(nb_workers = nb_workers, tile_mode = tile_mode) as eoscale_manager:

        img_1 = eoscale_manager.open_raster(raster_path = input_img_path)

        stats = eoexe.n_images_to_m_scalars(inputs = [img_1],
                                            image_filter = stats_filter,
                                            nb_output_scalars = 2,
                                            concatenate_filter = stats_concatenate,
                                            context_manager = eoscale_manager,
                                            filter_desc= "Min value processing...")

        print(stats[0], stats[1])

    #############################################################################################################################
    # All shared resources are automatically released
