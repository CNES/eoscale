import numpy as np

from eoscale.data_types import VirtualPath
from eoscale.eo_executors import n_images_to_m_scalars
from eoscale.manager import EOContextManager



def stats_filter(input_buffers: list,
                 input_profiles: list,
                 filter_parameters: dict) -> list[float]:
    """
    Computes statistics (minimum and maximum) for each input buffer.

    Parameters
    ----------
    input_buffers : list[np.ndarray]
        List of input numpy arrays, each representing an image buffer.

    input_profiles : list[dict]
        List of input profiles associated with each buffer.

    filter_parameters : dict
        Additional parameters for the statistics computation.

    Returns
    -------
    list[float]
        List containing the minimum and maximum values computed across all input buffers.
    """
    all_min = [np.min(img) for img in input_buffers]
    all_max = [np.max(img) for img in input_buffers]
    return [np.min(all_min), np.max(all_max)]


def stats_concatenate(output_scalars, chunk_output_scalars, tile) -> None:
    """
    Concatenates statistics (minimum and maximum) from chunk computations to the overall output.

    Parameters
    ----------
    output_scalars : list[float]
        Overall statistics output, containing minimum and maximum values.

    chunk_output_scalars : list[float]
        Statistics computed from a chunk of data, containing minimum and maximum values.

    tile : Any
        Additional data associated with the chunk computation.
    """
    output_scalars[0] = min(output_scalars[0], chunk_output_scalars[0])
    output_scalars[1] = max(output_scalars[1], chunk_output_scalars[1])


def minmax_filter(context: EOContextManager,
                  inputs: list[str] | list[VirtualPath],
                  ) -> tuple[float, float]:
    """
    Minimum and maximum values from a list of raster inputs using n_images_to_m_scalars.

    Parameters
    ----------
    context : EOContextManager
        EOContextManager instance for handling raster operations and contexts.

    inputs : list[str] | list[VirtualPath]
        List of paths or VirtualPath instances representing input raster data.

    Returns
    -------
    tuple[float, float]
        Tuple containing the computed minimum and maximum values from the input rasters.
    """
    imgs = [context.open_raster(raster_path=img) for img in inputs]
    return n_images_to_m_scalars(inputs=imgs,
                                  image_filter=stats_filter,
                                  nb_output_scalars=2,
                                  concatenate_filter=stats_concatenate,
                                  context_manager=context,
                                  filter_desc="Min Max values processing...")
