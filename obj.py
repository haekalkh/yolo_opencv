import cv2
from ultralytics import YOLO
import numpy as np

#frame 
frameX = 620
frameY = 480

centerFrameX = frameX // 2
centerFrameY = frameY // 2

model = YOLO('best_1123.pt')
className = ["matang", "agak matang", "mentah"]

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, frameX)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frameY)

while True:
    ret, frame = cam.read()

    if not ret:
        print("Error")
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bboxCenterX = (x1 + x2) // 2
            bboxCenterY = (y1 + y2) // 2
            relX = bboxCenterX - centerFrameX
            relY = bboxCenterY - centerFrameY

            labelId = int(box.cls)
            confidence = box.conf.item()
            label = className[labelId] 

            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f'{relX} {relY}', (x1, y1 - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.circle(frame, (frameX, frameY), 5, (0, 0, 255), -1)
        cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows() 