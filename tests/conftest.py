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

import os
from dataclasses import dataclass

import numpy as np
import pytest
from pathlib import Path

from tests.utils import create_raster


@dataclass
class EOScaleTestsData():
    project_dir: Path

    def __post_init__(self):
        self.dsm_raster = self.project_dir / "examples" / "data" / "dsm.tif"


@pytest.fixture
def eoscale_paths(request) -> EOScaleTestsData:
    return EOScaleTestsData(Path(__file__).resolve().parent.parent)


@pytest.fixture(scope="function")
def raster_data_generator(request, tmpdir) -> str:
    """
    Fixture that generates a temporary raster file from a numpy array and provides its path.

    This fixture creates a temporary GeoTIFF raster file from a numpy array specified
    in the test parameters (`request.param`). The raster file is deleted after the test
    completes.

    Parameters
    ----------
    request : _pytest.fixtures.FixtureRequest
        Pytest request object representing the test request.

    tmpdir : py.path.local
        Pytest fixture providing a temporary directory path.

    Yields
    ------
    str
        Path to the temporary raster file.

    Example
    -------
    Use this fixture in a test function to obtain a path to the temporary raster file:

    >>> def test_raster_processing(raster_data_generator):
    >>>     # `raster_data_generator` is the path to the temporary raster file
    >>>     assert os.path.isfile(raster_data_generator)
    >>>     # Perform tests with the raster data
    >>>     ...

    """
    data = request.param
    tmp_data_file = tmpdir / "data.tif"
    create_raster(data, str(tmp_data_file))
    yield str(tmp_data_file)
    os.remove(str(tmp_data_file))