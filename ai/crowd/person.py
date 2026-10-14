"""Single-frame person detection only. No tracking, identity or package download."""
import math
import os
from pathlib import Path
from pydantic import model_validator
from ai.core.analysis import Contract, Box, Score
from ai.inference.coordinates import normalize_box, validate_image_size

MODEL_VERSION = 'yolo26n-person-v1.0.0'
INFERENCE_VERSION = 'person-resolution-v1.1.0'


class CrowdFailure(Exception):
    def __init__(self,code):
        self.code=code
        super().__init__(code)


class PersonDetection(Contract):
    bbox: Box
    confidence: Score

    @model_validator(mode='after')
    def ordered(self):
        if self.bbox[0]>=self.bbox[2] or self.bbox[1]>=self.bbox[3]:
            raise ValueError('Invalid person bbox')
        return self


class PersonDetector:
    def __init__(self,path,device='cpu',loader=None,confidence=.25,max_detections=1000):
        self.path,self.device,self.loader=Path(path),device,loader
        if not 0<=confidence<=1 or not math.isfinite(confidence) or type(max_detections) is not int or max_detections<1:
            raise ValueError('Invalid person detector configuration')
        self.confidence,self.max_detections=confidence,max_detections
        self.model=None

    def start(self):
        if self.model is not None:
            return
        try:
            if not self.path.is_file():
                raise ValueError('Local person weights unavailable')
            loader=self.loader
            if loader is None:
                os.environ['YOLO_AUTOINSTALL']='false'
                from ultralytics import YOLO
                loader=YOLO
            model=loader(str(self.path))
            names=dict(enumerate(model.names)) if isinstance(model.names,list) else model.names
            ids=[index for index,label in names.items() if label=='person']
            if len(ids)!=1 or type(ids[0]) is not int:
                raise ValueError('Expected a person class')
            self.person_id=ids[0]
            self.model=model
        except Exception:
            raise CrowdFailure('PERSON_MODEL_LOAD_FAILED') from None

    def detect(self,image,image_size=640):
        if type(image_size) is not int or image_size not in (640,960,1280,1600):
            raise ValueError('Unsupported person inference image size')
        if self.model is None:
            raise CrowdFailure('PERSON_MODEL_NOT_READY')
        try:
            results=self.model.predict(source=image,classes=[self.person_id],device=self.device,
                conf=self.confidence,imgsz=image_size,max_det=self.max_detections,verbose=False)
        except Exception:
            raise CrowdFailure('PERSON_INFERENCE_FAILED') from None
        try:
            if len(results)!=1:
                raise ValueError('Expected one image')
            result=results[0]
            validate_image_size(image.size,result.orig_shape)
            if len(result.boxes)>=self.max_detections:
                raise CrowdFailure('PERSON_DETECTION_LIMIT')
            detections=[]
            for box in result.boxes:
                index=box.cls.item()
                if isinstance(index,bool) or not math.isfinite(index) or index!=int(index):
                    raise ValueError('Invalid class index')
                if index!=self.person_id:
                    continue
                detections.append(PersonDetection(bbox=normalize_box(box.xyxy[0].tolist(),image.size),confidence=box.conf.item()))
            return sorted(detections,key=lambda d:(d.bbox,-d.confidence))
        except CrowdFailure:
            raise
        except Exception:
            raise CrowdFailure('INVALID_PERSON_OUTPUT') from None
