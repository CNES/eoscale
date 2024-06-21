from dataclasses import dataclass

import pytest
from pathlib import Path


@dataclass
class EOScaleTestsData():
    project_dir: Path

    def __post_init__(self):
        self.dsm_raster = self.project_dir / "examples" / "data" / "dsm.tif"


@pytest.fixture
def eoscale_paths(request) -> EOScaleTestsData:
    return EOScaleTestsData(Path.cwd().parent.parent)
