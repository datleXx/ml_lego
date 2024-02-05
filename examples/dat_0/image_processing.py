import cv2 
from ultralytics import YOLO
import random 
from aiko_services import aiko

# MODEL = YOLO("weights/yolov8n.pt", "v8")

def generate_colors(): 
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

def load_image(filepath): 
    image = cv2.imread(filepath)

    return image

def open_image(image): 
    if image is None: 
        print("Error: Cannot find image to open ! Try to load the image first ... ")
        return None
    cv2.imshow("Object", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def model_inference(image, model):
    if image is not None: 
        detect_params = model.predict(source=[image], conf=0.45, save=False)
        prediction = detect_params[0].boxes
        return prediction
    else: 
        print("Error: Cannot find image to do model inference ! TRy to load the image first ... ")
        return None 
    
def draw_box(image, prediction, detection_colors, class_list): 
    if image is not None and prediction is not None: 
        for box in prediction: 
            classID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            boundingbox = box.xyxy.numpy()[0]

            cv2.rectangle(
                image,
                (int(boundingbox[0]), int(boundingbox[1])),
                (int(boundingbox[2]), int(boundingbox[3])),
                detection_colors[int(classID)],
                3,
            )

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                image,
                class_list[int(classID)] + " " + str(round(conf, 3)) + "%",
                (int(boundingbox[0]), int(boundingbox[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )        
        return image 
    print("Error: Image not loaded, or haven't perform model inference on the image")
    return None 

