from unittest.mock import patch

import numpy as np
from scipy.ndimage import generic_filter

from eoscale.eo_executors import compute_mp_tiles
from eoscale.filters.concatenate_images import concatenate_images
from eoscale.filters.generic_kernel import generic_kernel_filter
from eoscale.manager import EOContextManager
from tests.utils import read_raster


def test_n_to_m_imgs(eoscale_paths):
    """
    """
    imgs = [eoscale_paths.dsm_raster, eoscale_paths.dsm_raster]
    with EOContextManager(nb_workers=4, tile_mode=True) as eoscale_manager:
        concatenate_vpath = concatenate_images(eoscale_manager, imgs)
        concatenate_array = eoscale_manager.get_array(concatenate_vpath)
        assert concatenate_array.shape == (len(imgs), 512, 512)


def compute_mp_tiles_margin_0(inputs: list,
                              stable_margin: int,
                              nb_workers: int,
                              tile_mode: bool,
                              context_manager: EOContextManager):
    stable_margin = 0
    return compute_mp_tiles(inputs,
                            stable_margin,
                            nb_workers,
                            tile_mode,
                            context_manager)


def constant(array: np.ndarray, constant_value: int):
    return constant_value


def test_generic_filter_constant(eoscale_paths):
    """
    """
    with EOContextManager(nb_workers=4, tile_mode=True) as eoscale_manager:
        out_vpath = generic_kernel_filter(eoscale_manager,
                                          [eoscale_paths.dsm_raster, eoscale_paths.dsm_raster],
                                          constant, 2, func_kwarg={"constant_value": 42})[0]
        arr_const = eoscale_manager.get_array(out_vpath)
        assert arr_const.shape == (1, 512, 512)
        print(np.unique(arr_const, return_counts=True))


def test_n_to_m_imgs_margin(eoscale_paths):
    """
    """
    with EOContextManager(nb_workers=4, tile_mode=True) as eoscale_manager:
        out_vpath = generic_kernel_filter(eoscale_manager,
                                          [eoscale_paths.dsm_raster, eoscale_paths.dsm_raster],
                                          np.sum, 2)[0]
        arr_margin = eoscale_manager.get_array(out_vpath)
        with patch('eoscale.eo_executors.compute_mp_tiles', new=compute_mp_tiles_margin_0):
            out_vpath_no_marge = generic_kernel_filter(eoscale_manager,
                                                       [eoscale_paths.dsm_raster,
                                                        eoscale_paths.dsm_raster],
                                                       np.sum, 2)[0]
            arr_no_margin = eoscale_manager.get_array(out_vpath_no_marge)
        assert arr_margin.shape == arr_no_margin.shape, "results with/without margin must have the same shape"
        assert np.allclose(arr_margin, arr_no_margin) is False, "results with/without margin must be different"
        arr_margin = np.copy(arr_margin)
    arr_ref = generic_filter(read_raster(eoscale_paths.dsm_raster), np.sum, size=(1, 5, 5), mode="constant", cval=0)
    assert np.allclose(arr_ref, arr_margin), "kernel processing different from reference"
