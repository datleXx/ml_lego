import cv2
import time

cap = cv2.VideoCapture(0)
test = cap 
if not cap.isOpened(): 
    print("Error")
    exit()

while True : 
    ret, frame = cap.read()
    if not ret: 
        print("End of stream ? Exitting ... ")
        break
    cv2.imshow("Obj_detect",frame)
    # time.sleep(0.1)
    if cv2.waitKey(1) == ord("q"): 
        break
cap.release()
cv2.destroyAllWindows()