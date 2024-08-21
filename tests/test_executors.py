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

from unittest.mock import patch

import numpy as np
import rasterio
from scipy.ndimage import generic_filter
import pytest

from eoscale.eo_executors import compute_mp_tiles
from eoscale.filters.concatenate_images import concatenate_images
from eoscale.filters.generic_kernel import generic_kernel_filter
from eoscale.filters.stats_minmax import minmax_filter
from eoscale.manager import EOContextManager
from tests.utils import read_raster, assert_profiles


def test_concatenate_path_virtual_path(eoscale_paths):
    """
    Tests the concatenation of multiple images from disk and VirtualPath then verifies the shape of
    the resulting array.

    This test verifies that the `concatenate_images` function correctly concatenates multiple
    input raster images into a single array and checks that the resulting array has the
    expected shape.

    Parameters
    ----------
    eoscale_paths : EOScaleTestsData
        An instance of EOSScalePaths providing the paths to the DSM raster images used as
        inputs in the test.

    Raises
    ------
    AssertionError
        If the shape of the concatenated array does not match the expected shape and if bands are
        not equal, indicating a failure in the concatenation process.
    """
    with rasterio.open(eoscale_paths.dsm_raster, "r") as raster_dataset:
        profile = raster_dataset.profile
        data = raster_dataset.read()

    with EOContextManager(nb_workers=4, tile_mode=True) as eoscale_manager:
        new_key = eoscale_manager.create_memview(key="access_key", arr_subset=data, arr_subset_profile=profile)
        imgs = [eoscale_paths.dsm_raster, new_key]
        concatenate_vpath = concatenate_images(eoscale_manager, imgs)
        concatenate_array = eoscale_manager.get_array(concatenate_vpath)
        assert concatenate_array.shape == (len(imgs), 512, 512), "concatenate filter fails"
        assert np.allclose(concatenate_array[0, :, :], concatenate_array[1, :, :])


def compute_mp_tiles_margin_0(inputs: list,
                              stable_margin: int,
                              nb_workers: int,
                              tile_mode: bool,
                              context_manager: EOContextManager):
    """Patch to force stable_margin to 0"""
    stable_margin = 0
    return compute_mp_tiles(inputs,
                            stable_margin,
                            nb_workers,
                            tile_mode,
                            context_manager)


def constant(array: np.ndarray, constant_value: int):
    return constant_value


@pytest.mark.parametrize(
    "expected_type", [np.float32, np.uint8]
)
def test_constant(expected_type, eoscale_paths):
    """
    Tests the generic kernel filter with a constant function and different output types.

    This test applies a kernel filter with a constant function to input raster images and
    verifies that the output type matches the expected type and do not contains any new values.
    It also checks that the resulting array has the correct shape and that the values are as
    expected.

    Parameters
    ----------
    expected_type : type
        The expected data type for the output array, parameterized to test both `np.float32` and `np.uint8`.
    eoscale_paths : EOScaleTestsData
        An instance of EOScaleTestsData providing the paths to the DSM raster images used as inputs in the test.

    Raises
    ------
    AssertionError
        If any of the assertions fail, indicating discrepancies in the output type, shape, or values of
        the processed array.
    """
    const_value = 42
    with EOContextManager(nb_workers=4, tile_mode=True) as eoscale_manager:
        out_vpath = generic_kernel_filter(eoscale_manager,
                                          [eoscale_paths.dsm_raster, eoscale_paths.dsm_raster],
                                          constant, 2, dtype=expected_type, func_kwarg={"constant_value": const_value})[
            0]
        arr_const = eoscale_manager.get_array(out_vpath)
        assert arr_const.dtype == expected_type, "wrong output type"
        assert arr_const.shape == (1, 512, 512)
        counts = np.unique(arr_const, return_counts=True)
        assert counts[0][0] == const_value and counts[-1][0] == 512 * 512, "margin introduce unexpected values"


def test_n_to_m_imgs_margin(eoscale_paths, tmpdir):
    """
    Test the generic kernel filter with and without margins and verify the results.

    This test applies a kernel filter with a summation function to input raster images,
    once considering margins and once without considering margins. It ensures that the
    results differ when margins are included or excluded and checks if the kernel processing
    is consistent with a reference implementation. Check also if every rasterio profile
    are consistent.

    Parameters
    ----------
    eoscale_paths : EOScaleTestsData
        An instance of EOSScalePaths providing the paths to the DSM raster images used
        as inputs in the test.

    tmpdir : py.path.local
        Pytest fixture providing a temporary directory path.

    Raises
    ------
    AssertionError
        If any of the assertions fail, indicating discrepancies in the shapes or values of
        the arrays processed with and without margins or inconsistencies with the reference
        implementation.
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

        raster_margin_file = tmpdir / "raster_margin.tif"
        raster_no_margin_file = tmpdir / "raster_no_margin.tif"
        eoscale_manager.write(key=out_vpath, img_path=str(raster_margin_file))
        eoscale_manager.write(key=out_vpath_no_marge, img_path=str(raster_no_margin_file))

    arr_ref = generic_filter(read_raster(eoscale_paths.dsm_raster), np.sum, size=(1, 5, 5), mode="constant", cval=0)
    assert np.allclose(arr_ref, arr_margin), "kernel processing different from reference"
    assert assert_profiles([str(eoscale_paths.dsm_raster),
                            str(raster_margin_file),
                            str(raster_no_margin_file)]), "profiles must be consistent with input data"


@pytest.mark.parametrize("raster_data_generator", [np.expand_dims(np.ones((512, 512)) * 50000, axis=0),
                                                   np.expand_dims(np.ones((512, 512)) * -50000, axis=0),
                                                   np.expand_dims(np.random.random((512, 512)), axis=0)], indirect=True)
def test_n_images_m_scalars(raster_data_generator, eoscale_paths):
    """
    Test function for processing scalar with EOContextManager.

    This test verifies the functionality of processing scalar numpy arrays
    alongside a DSM raster (containing values [-32768.0, 124.62529]) using an EOContextManager
    instance. It parametrizes over different scalar arrays: large positive values, large negative
    values, and random values [0.0, 1.0).

    Parameters
    ----------
    raster_data_generator : str
        The scalar numpy array provided by the pytest fixture. This array is
        expanded to have a shape of (bands, height, width).

    eoscale_paths : EOScaleTestsData
        Fixture providing paths to data used in the test.

    Raises
    ------
    AssertionError
        If the computed minimum or maximum values from the processed data do not
        match the expected values derived from the input numpy_data and DSM raster.
    """
    dsm_min = -32768.0
    dsm_max = 124.62529

    test_arr = read_raster(raster_data_generator)
    expected_min = min(np.min(test_arr), dsm_min)
    expected_max = max(np.max(test_arr), dsm_max)

    with EOContextManager(nb_workers=4, tile_mode=True) as eoscale_manager:
        min_max = minmax_filter(eoscale_manager, [eoscale_paths.dsm_raster, raster_data_generator])
        assert len(min_max) == 2, "minmax_filter output must be a list of 2 elements"
        test_min, test_max = min_max
        assert np.allclose(test_min, expected_min), "minmax_filter fail to detect the minimum"
        assert np.allclose(test_max, expected_max), "minmax_filter fail to detect the maximum"
