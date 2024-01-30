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
from aiko_services import ImageCV

_LOGGER = aiko.logger(__name__)

# --------------------------------------------------------------------------- #

class PE_GetImage(PipelineElement):
    def __init__(self, context):
        context.set_protocol("getimage")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, file_path:str) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_GetImage: {context}, in file_path: {file_path}")
        image_object = ImageCV()
        image_object.load_path(file_path)
        _LOGGER.info(f"PE_GetImage: {context}, out image_object: {image_object}")
        return True, {"image_object": image_object}

# --------------------------------------------------------------------------- #

class PE_ModelInference(PipelineElement):
    def __init__(self, context):
        context.set_protocol("modelinference")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, image_object) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_ModelInference: {context}, in image_object: {image_object}")
        image_object.model_inference()
        _LOGGER.info(f"PE_ModelInference: {context}, out image_object: {image_object}")
        return True, {"image_object": image_object}

# --------------------------------------------------------------------------- #
    
class PE_DrawBoundingBox(PipelineElement):
    def __init__(self, context):
        context.set_protocol("boundingbox")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, image_object) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_DrawBoundingBox: {context}, in image_object: {image_object}")
        image_object.draw_bounding_box()
        _LOGGER.info(f"PE_DrawBoundingBox: {context}, out image_object: {image_object}")
        return True, {"image_object": image_object}

# --------------------------------------------------------------------------- #
    
class PE_DisplayImage(PipelineElement):
    def __init__(self, context):
        context.set_protocol("displayimage")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, image_object) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_DisplayImage: {context}, in image_object: {image_object}")
        image_object.open_image()
        _LOGGER.info(f"PE_DisplayImage: {context}, out image_status: {image_object}")
        return True, {"image_object_status": image_object}