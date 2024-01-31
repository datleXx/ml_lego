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
from aiko_services import VideoCV

import cv2
import time

_LOGGER = aiko.logger(__name__)
WIDTH = 0
HEIGHT = 0 
FPS = 0
OUTPUT_PATH = None 
# --------------------------------------------------------------------------- #

class PE_GetVideo(PipelineElement):
    def __init__(self, context):
        context.set_protocol("getvideo")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, file_path:str) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_GetImage: {context}, in file_path: {file_path}")
        video_object = VideoCV()
        video_object.load_path(file_path)
        video_frames = video_object.get_frame()
        WIDTH = video_object.width 
        HEIGHT = video_object.height 
        FPS = video_object.fps
        OUTPUT_PATH = video_object.output_path
        _LOGGER.info(f"PE_GetVideo: {context}, out video_frames: {video_frames}")
        return True, {"video_frames": video_frames,
                      "WIDTH": WIDTH, 
                      "HEIGHT": HEIGHT, 
                      "OUTPUT_PATH": OUTPUT_PATH,
                      "FPS": FPS}

# --------------------------------------------------------------------------- #

class PE_ModelInference(PipelineElement):
    def __init__(self, context):
        context.set_protocol("modelinference")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, video_frames, WIDTH, HEIGHT, OUTPUT_PATH, FPS) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_ModelInference: {context}, in video_frames: {video_frames}")
        new_video_frames = []
        skip_frame = 3
        for i, frame in enumerate(video_frames): 
            if i % skip_frame == 0: 
                image_obj = ImageCV()
                image_obj.tensor = frame
                image_obj.model_inference()
                image_obj.draw_bounding_box()
                new_video_frames.append(image_obj)
        _LOGGER.info(f"PE_ModelInference: {context}, out video_frames: {new_video_frames}")
        return True, {"video_frames": new_video_frames,
                      "WIDTH": WIDTH, 
                      "HEIGHT": HEIGHT, 
                      "OUTPUT_PATH": OUTPUT_PATH,
                      "FPS": FPS}
# --------------------------------------------------------------------------- #
    
# class PE_DrawBoundingBox(PipelineElement):
#     def __init__(self, context):
#         context.set_protocol("boundingbox")
#         context.get_implementation("PipelineElement").__init__(self, context)

#     def process_frame(self, context, video_frames, WIDTH, HEIGHT, OUTPUT_PATH, FPS) -> Tuple[bool, dict]:
#         _LOGGER.info(f"PE_DrawBoundingBox: {context}, in video_frames: {video_frames}")
#         new_video_frames = []
#         for image_obj in video_frames:
#             image_obj.draw_bounding_box()
#             new_video_frames.append(image_obj)

#         _LOGGER.info(f"PE_DrawBoundingBox: {context}, out video_frames: {new_video_frames}")
#         return True, {"video_frames": new_video_frames,
#                       "WIDTH": WIDTH, 
#                       "HEIGHT": HEIGHT, 
#                       "OUTPUT_PATH": OUTPUT_PATH,
#                       "FPS": FPS}

# --------------------------------------------------------------------------- #
class PE_SaveVideo(PipelineElement):
    def __init__(self, context):
        context.set_protocol("savevideo")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, video_frames, WIDTH, HEIGHT, OUTPUT_PATH, FPS) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_SaveVideo: {context}, in video_frames: {video_frames}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))
        for image_obj in video_frames:  
            out.write(image_obj.tensor)

        # cv2.destroyAllWindows()
        _LOGGER.info(f"PE_SaveVideo: {context}, out video_path: {OUTPUT_PATH}")
        return True, {
            "OUTPUT_PATH": OUTPUT_PATH
                      }
# --------------------------------------------------------------------------- #
class PE_DisplayVideo(PipelineElement):
    def __init__(self, context):
        context.set_protocol("displayvideo")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, OUTPUT_PATH) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_DisplayVideo: {context}, in video_path: {OUTPUT_PATH}")
        cap = cv2.VideoCapture(OUTPUT_PATH)
        if not cap.isOpened(): 
            print("Cannot open video...")
            exit()
        while True: 
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            cv2.imshow("video", frame)
            time.sleep(0.1)
            if cv2.waitKey(1) == ord("q"): 
                break
        cap.release()
        cv2.destroyAllWindows()

        # cv2.destroyAllWindows()
        _LOGGER.info(f"PE_DisplayVideo: {context}, out video_path: {OUTPUT_PATH}")
        return True, {"video_path": {OUTPUT_PATH}}