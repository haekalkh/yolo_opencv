import numpy as np
import cv2
from ultralytics import YOLO

def detect_object():
    # Load YOLO model
    model = YOLO('best_1123.pt')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 620)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

                # Konversi ke koordinat global (misalnya 1 pixel = 1 mm)
                scale_factor = 1
                x = bbox_center_x * scale_factor
                y = bbox_center_y * scale_factor
                z = 0  # Sesuaikan jika diperlukan

                cap.release()
                cv2.destroyAllWindows()
                return np.array([x, y, z])

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    object_position = detect_object()
    if object_position is not None:
        print(f"Detected object position: {object_position}")
