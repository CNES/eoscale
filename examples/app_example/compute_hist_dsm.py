import numpy as np
import eoscale.manager as eom
import eoscale.eo_executors as eoexe

def min_max_filter(input_buffers : list, 
                 input_profiles: list, 
               filter_parameters: dict):

    """ """
    img = input_buffers[0]
    img[img == input_profiles[0]['nodata']] = np.nan
    min_value : float = np.nanmin(img)
    max_value : float = np.nanmax(img)

    return [min_value, max_value]

def min_max_concatenate(output_scalars, chunk_output_scalars, tile):
  output_scalars[0] = min(output_scalars[0], chunk_output_scalars[0] )
  output_scalars[1] = max(output_scalars[1], chunk_output_scalars[1] )

def hist_concatenate(output_scalars, chunk_output_scalars, tile):
    output_scalars[0] += chunk_output_scalars[0]


def hist_filter(input_buffers : list, 
                 input_profiles: list, 
               filter_parameters: dict):
    dsm = input_buffers[0][0]
    dsm_wo_nan = dsm[~np.isnan(dsm)]
    return np.histogram(dsm_wo_nan, filter_parameters['nb_bins'])[0]


if __name__ == "__main__":

    input_image_path: str = "/work/EOLAB/DATA/SNCF/2023/Nice_ai4geo/MNS/crop_width_large_mccnn/dsm.tif"
    output_path: str = "/work/scratch/data/lallemd/test_eoscale/hist.npy"

    precision_alti = 2.0
    nb_workers: int = 8
    tile_mode: bool = True

    with eom.EOContextManager(nb_workers = nb_workers, tile_mode = tile_mode) as eoscale_manager:

      dsm = eoscale_manager.open_raster(raster_path = input_image_path)

      [dsm_min, dsm_max] = eoexe.n_images_to_m_scalars(inputs = [dsm],
                                            image_filter = min_max_filter,
                                            nb_output_scalars = 2,
                                            concatenate_filter = min_max_concatenate,
                                            context_manager = eoscale_manager,
                                            filter_desc= "Min/Max value processing...")

      dict_bins = {'min_Z' : dsm_min, 'nb_bins' : int((dsm_max - dsm_min)/precision_alti)}
      stats = eoexe.n_images_to_m_scalars(inputs = [dsm],
                                          filter_parameters = dict_bins,
                                          image_filter = hist_filter,
                                          nb_output_scalars = 1,
                                          output_scalars = [np.zeros(dict_bins["nb_bins"], dtype=np.float64)],
                                          concatenate_filter = hist_concatenate,
                                          context_manager = eoscale_manager,
                                          filter_desc= "Compute histogram...")

      np.save(output_path, stats[0])



