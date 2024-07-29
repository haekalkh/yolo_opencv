import cv2
from ultralytics import YOLO
import time
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO('best_1123.pt')
classNames = ["Matang", "Agak Matang", "Mentah"]

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get the frame dimensions
frame_width = 640
frame_height = 480

# Calculate the center of the frame
frame_center_x = frame_width // 2
frame_center_y = frame_height // 2

# Variables for FPS calculation
prev_time = time.time()
fps = 0

# Initialize Kalman filter parameters
kalman_filters = {}

def get_kalman_filter(id):
    if id not in kalman_filters:
        kalman_filter = cv2.KalmanFilter(4, 2)
        kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32) * 0.03
        kalman_filter.measurementNoiseCov = np.array([[1, 0],
                                                     [0, 1]], np.float32) * 1
        kalman_filter.errorCovPost = np.eye(4, dtype=np.float32)
        kalman_filters[id] = kalman_filter
    return kalman_filters[id]

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

# Initialize PID controllers for J3, J4, and J5 directions
pid_J3 = PID(P=0.1, I=0.01, D=0.01)
pid_J4 = PID(P=0.1, I=0.01, D=0.01)
pid_J5 = PID(P=0.1, I=0.01, D=0.01)

# Store corrected coordinates for visualization
corrected_J3_list = []
corrected_J4_list = []
corrected_J5_list = []

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                bbox_center_x = (x1 + x2) // 2
                bbox_center_y = (y1 + y2) // 2
                rel_x = bbox_center_x - frame_center_x
                rel_y = bbox_center_y - frame_center_y

                # Get the depth value at the center of the bounding box
                bbox_center_z = depth_frame.get_distance(bbox_center_x, bbox_center_y)

                label_id = int(box.cls)
                confidence = box.conf.item()
                label = classNames[label_id] if label_id < len(classNames) else f'Class {label_id}'

                # Kalman filter
                kalman_filter = get_kalman_filter(label_id)
                measurement = np.array([bbox_center_x, bbox_center_y], np.float32)
                kalman_filter.correct(measurement)
                prediction = kalman_filter.predict()
                pred_x, pred_y = int(prediction[0]), int(prediction[1])

                # PID Controller
                correction_J3 = pid_J3.compute(frame_center_x, bbox_center_x)
                correction_J4 = pid_J4.compute(frame_center_y, bbox_center_y)
                correction_J5 = pid_J5.compute(0, bbox_center_z)  # Assuming setpoint for J5 is 0
                corrected_J3 = int(frame_center_x + correction_J3)
                corrected_J4 = int(frame_center_y + correction_J4)
                corrected_J5 = bbox_center_z + correction_J5  # Simulate correction for J5

                # Store corrected coordinates
                corrected_J3_list.append(corrected_J3)
                corrected_J4_list.append(corrected_J4)
                corrected_J5_list.append(corrected_J5)

                # Print corrected coordinates
                print(f'Corrected Coordinates: J3={corrected_J3}, J4={corrected_J4}, J5={corrected_J5:.2f}')

                # Draw bounding box
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label, confidence, and relative coordinates
                cv2.putText(color_image, f'{label} {confidence:.2f}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
                cv2.putText(color_image, f'({rel_x}, {rel_y})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

                # Draw the predicted center
                cv2.circle(color_image, (pred_x, pred_y), 5, (0, 0, 255), -1)

                # Draw the corrected center
                cv2.circle(color_image, (corrected_J3, corrected_J4), 5, (255, 0, 0), -1)

        cv2.circle(color_image, (frame_center_x, frame_center_y), 5, (0, 0, 255), -1)
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(color_image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Webcam', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

    # Plot the kinematics
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(corrected_J3_list, label='J3')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(corrected_J4_list, label='J4')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(corrected_J5_list, label='J5')
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('Correction Value')
    plt.suptitle('Kinematics of J3, J4, and J5')
    plt.tight_layout()
    plt.show()
