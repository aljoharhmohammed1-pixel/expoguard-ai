import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    app_env: Literal['development', 'production', 'test'] = 'development'
    version: str = Field(default='0.1.0-integration', min_length=1)
    model_path: Path = ROOT/'ai/models/infernoguard_best.pt'
    person_model_path: Path = ROOT/'ai/models/yolo26n.pt'
    camera_config_path: Path = ROOT/'config/cameras.json'
    include_person_detections: bool = False
    device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    max_image_mb: int = Field(default=10, ge=1, le=20)
    playbook_dir: Path = ROOT/'playbooks'

    @classmethod
    def from_env(cls):
        mapping = {'APP_ENV':'app_env', 'AI_SERVICE_VERSION':'version', 'FIRE_MODEL_PATH':'model_path',
                   'MODEL_DEVICE':'device', 'MAX_IMAGE_SIZE_MB':'max_image_mb', 'PLAYBOOK_DIR':'playbook_dir',
                   'PERSON_MODEL_PATH':'person_model_path', 'CAMERA_CONFIG_PATH':'camera_config_path',
                   'INCLUDE_PERSON_DETECTIONS':'include_person_detections'}
        return cls.model_validate({field:os.environ[key] for key,field in mapping.items() if key in os.environ})
