import hashlib
from pathlib import Path
from typing import Literal
from pydantic import Field, StrictBool, model_validator
from ai.core.analysis import Contract, Identifier, Score
from ai.crowd.geometry import validate_polygon


class CrowdThresholds(Contract):
    medium: Score = .5
    high: Score = .7
    critical: Score = .9

    @model_validator(mode='after')
    def ordered(self):
        if not self.medium < self.high < self.critical:
            raise ValueError('Require 0 <= medium < high < critical <= 1')
        return self


class CrowdConfig(Contract):
    enabled: StrictBool = False
    capacity: int | None = Field(default=None, strict=True, gt=0)
    roi: list[tuple[Score, Score]] | None = Field(default=None,min_length=3,max_length=64)
    useFullFrame: StrictBool = False
    thresholds: CrowdThresholds = Field(default_factory=CrowdThresholds)
    reviewMedium: StrictBool = False
    inferenceImageSize: Literal[640, 960, 1280, 1600] = 640

    @model_validator(mode='after')
    def geometry(self):
        if self.roi is not None:
            if self.useFullFrame:
                raise ValueError('Specify either ROI or full frame, not both')
            validate_polygon(self.roi)
        return self


class CameraConfig(Contract):
    cameraId: Identifier
    zoneId: Identifier | None = None
    crowd: CrowdConfig = Field(default_factory=CrowdConfig)

    @property
    def digest(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class CameraCatalog(Contract):
    cameras: list[CameraConfig] = Field(default_factory=list,max_length=10000)

    @model_validator(mode='after')
    def unique(self):
        if len({c.cameraId for c in self.cameras}) != len(self.cameras):
            raise ValueError('Duplicate camera ID')
        return self

    def resolve(self, camera_id):
        return next((c for c in self.cameras if c.cameraId==camera_id),None)


def load_cameras(path):
    try:
        with Path(path).open('rb') as stream:
            raw=stream.read(1024*1024+1)
        if len(raw)>1024*1024:
            raise ValueError('Camera catalog exceeds 1 MiB')
        return CameraCatalog.model_validate_json(raw)
    except (ValueError,OSError):
        raise ValueError('Invalid camera crowd configuration') from None
