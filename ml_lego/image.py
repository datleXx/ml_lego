import cv2 
from ultralytics import YOLO
import random 
from aiko_services import aiko

class ImageCV():
    def __init__(self): 
        self.image = None 
        self.detection_colors, self.class_list = self.generate_colors()
        self.boxes = None 
        

    def load_path(self, file_path): 
        self.image = cv2.imread(file_path)
        return self.image 
    
    def generate_colors(self): 
        # opening the file in read mode
        my_file = open("utils/coco.txt", "r")
        # reading the file
        data = my_file.read()
        # replacing end splitting the text | when newline ('\n') is seen.
        class_list = data.split("\n")
        my_file.close()

        # Generate random colors for class list
        detection_colors = []
        for i in range(len(class_list)):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            detection_colors.append((b, g, r))
        return detection_colors, class_list

    def model_inference(self, model): 
        model = model 
        if self.image is not None: 
            detect_params = model.predict(source=[self.image], conf=0.45, save=False)
            self.boxes = detect_params[0].boxes
            return self.boxes
        else: 
            print("Error: Cannot find image to do model inference ! TRy to load the image first ... ")
            return None 

    def draw_bounding_box(self): 
        if self.image is not None: 
            if self.boxes is not None: 

                for box in self.boxes: 
                    clsID = box.cls.numpy()[0]
                    conf = box.conf.numpy()[0]
                    bb = box.xyxy.numpy()[0]

                    cv2.rectangle(
                        self.image, 
                        (int(bb[0]), int(bb[1])),
                        (int(bb[2]), int(bb[3])),
                        self.detection_colors[int(clsID)],
                        3,    
                    )

                    # Display class name and confidence
                    font = cv2.FONT_HERSHEY_COMPLEX
                    cv2.putText(
                        self.image,
                        self.class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                        (int(bb[0]), int(bb[1]) - 10),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )
    
            print("Error: Haven't perform model inference on this image or haven't load the image ...") 
    def open_image(self): 
        if self.image is None: 
            print("Error: Cannot find image to open ! Try to load the image first ... ")
            return None
        cv2.imshow("Object", self.image)
        cv2.waitKey(1)

        
