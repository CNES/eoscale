import numpy as np
import pytest
import rasterio

from eoscale.filters.concatenate_images import concatenate_images
from eoscale.filters.generic_kernel import generic_kernel_filter
from eoscale.manager import EOContextManager


def test_release_memory(eoscale_paths):
    """
    Tests the release of memory associated with a concatenated raster image.

    Checks that the array does not own its data (indicating it is a view),
    and then releases the memory associated with the concatenated image. Finally, it verifies
    that attempting to access the array after releasing the memory raises a KeyError.

    Parameters
    ----------
    eoscale_paths : EoscalePaths
        Object containing paths to DSM raster images.
    """
    imgs = [eoscale_paths.dsm_raster, eoscale_paths.dsm_raster]
    with EOContextManager(nb_workers=4, tile_mode=True) as eoscale_manager:
        concatenate_vpath = concatenate_images(eoscale_manager, imgs)
        concatenate_array = eoscale_manager.get_array(concatenate_vpath)
        assert concatenate_array.flags["OWNDATA"] is False
        eoscale_manager.release(concatenate_vpath)
        with pytest.raises(KeyError):
            eoscale_manager.get_array(concatenate_vpath)


@pytest.mark.parametrize("raster_data_generator", [np.expand_dims(np.random.random((512, 512)), axis=0)], indirect=True)
def test_tile_mode(raster_data_generator):
    """
    Tests the behavior of the processing pipeline with different tile modes.

    This test compares the results of a generic kernel filter applied to a raster containing
    random values using two different EOContextManager instances: one with `tile_mode=True`
    and another with `tile_mode=False`. It verifies that the resulting arrays from both modes are
    identical by performing an element-wise comparison.

    Parameters
    ----------
    raster_data_generator : str
        The numpy array provided by the pytest fixture. This array is
        expanded to have a shape of (bands, height, width).
    """
    with EOContextManager(nb_workers=4, tile_mode=True) as eoscale_manager:
        vpath_tiled = generic_kernel_filter(eoscale_manager,
                                            [raster_data_generator],
                                            np.sum, 2)[0]
        arr_tiled = eoscale_manager.get_array(vpath_tiled).copy()
    with EOContextManager(nb_workers=5, tile_mode=False) as eoscale_manager:
        vpath_strips = generic_kernel_filter(eoscale_manager,
                                             [raster_data_generator],
                                             np.sum, 2)[0]
        arr_strips = eoscale_manager.get_array(vpath_strips).copy()
    assert np.allclose(arr_tiled, arr_strips), "results with tile_mode=True != tile_mode=False"


@pytest.mark.parametrize("raster_data_generator", [np.expand_dims(np.random.random((512, 512)), axis=0)], indirect=True)
def test_create_memview(raster_data_generator):
    """
    Test the ability to provide an array to EOContextManager and retrieve it.

    Parameters
    ----------
    raster_data_generator : str
        The numpy array provided by the pytest fixture. This array is
        expanded to have a shape of (bands, height, width).
    """
    with rasterio.open(raster_data_generator, "r") as raster_dataset:
        profile = raster_dataset.profile
        data = raster_dataset.read()

    with EOContextManager(nb_workers=4, tile_mode=True) as eoscale_manager:
        access_key = "some_access_key"
        new_key = eoscale_manager.create_memview(key=access_key, arr_subset=data, arr_subset_profile=profile)
        arr_from_memview = eoscale_manager.get_array(new_key)
        assert np.allclose(arr_from_memview, data), "EOContextManager.get_array method alter data coming from create_memview"
