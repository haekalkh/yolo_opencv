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

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame reading was not successful, break the loop
    if not ret:
        print("Error: Could not read frame")
        break

    # Perform detection
    results = model(frame)

    # Process and draw the detections on the frame
    for result in results:
        for box in result.boxes:
            # Accessing coordinates and other properties correctly
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to integers

            # Calculate the center of the bounding box
            bbox_center_x = (x1 + x2) // 2
            bbox_center_y = (y1 + y2) // 2

            # Calculate relative coordinates to the frame center
            rel_x = bbox_center_x - frame_center_x
            rel_y = bbox_center_y - frame_center_y

            label_id = int(box.cls)  # Class id
            confidence = box.conf.item()  # Confidence score

            # Get the label from classNames
            label = classNames[label_id] if label_id < len(classNames) else f'Class {label_id}'

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label, confidence, and relative coordinates
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
            cv2.putText(frame, f'({rel_x}, {rel_y})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    # Draw the center point of the frame
    cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 0, 255), -1)  # Red dot for center

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
