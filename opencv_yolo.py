import cv2
from ultralytics import YOLO
import time

# Load YOLO model
model = YOLO('best_1123.pt')
classNames = ["Matang", "Agak Matang", "Mentah"]

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Set frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 620)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get the frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the center of the frame
frame_center_x = frame_width // 2
frame_center_y = frame_height // 2

# Variables for FPS calculation
prev_time = time.time()
fps = 0

# Initialize PID controllers for x and y directions
class PID:
    def __init__(self, P=0.1, I=0.01, D=0.01):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, current):
        error = setpoint - current
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

pid_x = PID(P=0.1, I=0.01, D=0.01)
pid_y = PID(P=0.1, I=0.01, D=0.01)

def coord(bbox_center_x, bbox_center_y, frame_center_x, frame_center_y):
    # PID correction
    correction_x = pid_x.compute(frame_center_x, bbox_center_x)
    correction_y = pid_y.compute(frame_center_y, bbox_center_y)
    corrected_x = int(frame_center_x + correction_x)
    corrected_y = int(frame_center_y + correction_y)

    return corrected_x, corrected_y

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            bbox_center_x = (x1 + x2) // 2
            bbox_center_y = (y1 + y2) // 2

            # Call the coord function
            corrected_x, corrected_y = coord(bbox_center_x, bbox_center_y, frame_center_x, frame_center_y)

            # Print corrected coordinates
            print(f'Corrected Coordinates: X={corrected_x}, Y={corrected_y}')

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label and corrected coordinates
            label_id = int(box.cls)
            confidence = box.conf.item()
            label = classNames[label_id] if label_id < len(classNames) else f'Class {label_id}'
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
            cv2.putText(frame, f'Corrected: ({corrected_x}, {corrected_y})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

            # Draw the corrected center
            cv2.circle(frame, (corrected_x, corrected_y), 5, (255, 0, 0), -1)

    cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 0, 255), -1)
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
