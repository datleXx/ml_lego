import cv2, pickle
from ultralytics import YOLO

from aiko_services import aiko, PipelineElement

_LOGGER = aiko.logger(__name__)

model = YOLO("yolov8n.pt")

# --------------------------------------------------------------------------- #

class PE_ReadVideo(PipelineElement):
    def __init__(self, context):
        context.set_protocol("ReadVideo")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, video_file_name):
        _LOGGER.info(f"PE_ReadVideo: {context}, read an video")
        data = cv2.VideoCapture(video_file_name)

        success = 1
        video = []
        while success:
            success, image = data.read()
            video.append(image)

        return True, video

# --------------------------------------------------------------------------- #

class PE_Predict(PipelineElement):
    def __init__(self, context):
        context.set_protocol("predict")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, video):
        _LOGGER.info(f"PE_Predict: {context}, Model's prediction")
        results = model(source=video, show=True)

        return True

# --------------------------------------------------------------------------- #