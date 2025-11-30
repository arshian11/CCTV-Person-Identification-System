import cv2
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\devka\crowdhuman\yolov8n-face-lindevs.pt"

print("Loading YOLO Face Model...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

    # Draw
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
