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
def numpy_data(request, tmpdir) -> str:
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

    >>> def test_raster_processing(numpy_data):
    >>>     # `numpy_data` is the path to the temporary raster file
    >>>     assert os.path.isfile(numpy_data)
    >>>     # Perform tests with the raster data
    >>>     ...

    """
    data = request.param
    tmp_data_file = tmpdir / "data.tif"
    create_raster(data, str(tmp_data_file))
    yield str(tmp_data_file)
    os.remove(str(tmp_data_file))