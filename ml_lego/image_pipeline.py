# Usage
# ~~~~~
# aiko_pipeline create image_pipeline.json
#
# export AIKO_LOG_MQTT=false
# TOPIC=$NAMESPACE/$HOST/$PID/$SID/in
# mosquitto_pub -h $HOST -t $TOPIC -m "(process_frame (stream_id: 0) (file_path: 0))"
#
# To Do
# ~~~~~
# - None, yet !

from typing import Tuple

from aiko_services import aiko, PipelineElement
# from .image import ImageCV
from ultralytics import YOLO
from threading import Thread
import cv2
import copy
from .image_processing import *

import time

_LOGGER = aiko.logger(__name__)
MODEL = YOLO("weights/yolov8n.pt", "v8")
DETECTION_COLORS, CLASSLIST = generate_colors()
# --------------------------------------------------------------------------- #

class PE_GetImage(PipelineElement):
    def __init__(self, context):
        context.set_protocol("getimage")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, file_path:str) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_GetImage: {context}, in file_path: {file_path}")
        image = load_image(file_path)
        # image_object = load_path(file_path)
        _LOGGER.info(f"PE_GetImage: {context}, out image: {image}")
        return True, {"image": image}

# --------------------------------------------------------------------------- #

class PE_ModelInference(PipelineElement):
    def __init__(self, context):
        context.set_protocol("modelinference")
        context.get_implementation("PipelineElement").__init__(self, context)
        # self.model = YOLO("weights/yolov8n.pt", "v8")

    def process_frame(self, context, image) -> Tuple[bool, dict]:
        time_now = time.time()
        _LOGGER.debug(f"{self._id(context)}: ## TIME START: {time_now:0.3f} ##")
        # _LOGGER.info(f"PE_ModelInference: {context}, in image_object: {image_object}")
        prediction = model_inference(image, MODEL)
        time_now = time.time()
        _LOGGER.debug(f"{self._id(context)}: ## TIME END: {time_now:0.3f} ##")
        # _LOGGER.info(f"PE_ModelInference: {context}, out image_object: {image_object}")
        return True, {"prediction": prediction,
                      "image": image}

# --------------------------------------------------------------------------- #
    
class PE_DrawBoundingBox(PipelineElement):
    def __init__(self, context):
        context.set_protocol("boundingbox")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, prediction, image) -> Tuple[bool, dict]:
        time_now = time.time()
        _LOGGER.debug(f"{self._id(context)}: ## TIME: {time_now:0.3f} ##")
        # _LOGGER.info(f"PE_DrawBoundingBox: {context}, in image_object: {image_object}")
        image = draw_box(image, prediction, DETECTION_COLORS, CLASSLIST)
        # _LOGGER.info(f"PE_DrawBoundingBox: {context}, out image_object: {image_object}")
        return True, {"image": image}

# --------------------------------------------------------------------------- #
    
class PE_DisplayImage(PipelineElement):
    def __init__(self, context):
        context.set_protocol("displayimage")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, image) -> Tuple[bool, dict]:
        time_now = time.time()
        _LOGGER.debug(f"{self._id(context)}: ## TIME: {time_now:0.3f} ##")
        # _LOGGER.info(f"PE_DisplayImage: {context}, in image_object: {image_object}")
        open_image(image)
        # _LOGGER.info(f"PE_DisplayImage: {context}, out image_status: {image_object}")
        return True, {"image": image}

# --------------------------------------------------------------------------- #

class PE_GenerateFrame(PipelineElement):
    def __init__(self, context):
        context.set_protocol("generateframe")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, frame) -> Tuple[bool, dict]:
        time_now = time.time()
        _LOGGER.debug(f"{self._id(context)}: ## TIME: {time_now:0.3f} ##")
        return True, {"image": frame}
    
    def _run(self, context): 
        cap = cv2.VideoCapture(0)
        frame_id = 0
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while not context["terminate"]: 
            frame_context = copy.deepcopy(context)
        
            ret, frame = cap.read()
            if not ret: 
                print("Can't receive frame (stream end?). Exiting ...")
                break
            if frame_id % 5 == 0: 
                self.create_frame(frame_context, {"frame": frame})
            frame_id += 1
            # time.sleep(0.1)

        cap.release()
    
    def start_stream(self, context, stream_id):
        context["terminate"] = False
        Thread(target=self._run, args=(context,)).start()

    def stop_stream(self, context, stream_id):
        context["terminate"] = True