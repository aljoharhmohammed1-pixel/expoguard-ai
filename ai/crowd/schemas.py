import math
from typing import Literal
from pydantic import Field, StrictBool, model_validator
from ai.core.analysis import Contract, Identifier

CrowdLevel = Literal['NORMAL', 'MEDIUM', 'HIGH', 'CRITICAL']


class CrowdMetrics(Contract):
    peopleCount: int = Field(strict=True, ge=0)
    capacity: int | None = Field(default=None, strict=True, gt=0)
    occupancyRatio: float | None = Field(default=None, strict=True, ge=0, allow_inf_nan=False)
    crowdLevel: CrowdLevel | None = None
    cameraId: Identifier
    zoneId: Identifier | None = None
    roiApplied: StrictBool

    @model_validator(mode='after')
    def consistent(self):
        if self.capacity is None:
            if self.occupancyRatio is not None or self.crowdLevel is not None:
                raise ValueError('Unknown capacity cannot imply occupancy or level')
        elif self.occupancyRatio is None or self.crowdLevel is None or not math.isclose(
                self.occupancyRatio, self.peopleCount/self.capacity, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Occupancy must equal visible count / configured capacity')
        return self
