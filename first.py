# from ultralytics import YOLO
# model = YOLO("yolov8m_custom.pt")
# model.predict(source=0 , show=True, save=True, conf=0.5)

from ultralytics import YOLO
import cv2

# This code passes the real-time video captured by the webcam and sends it to yolo for prediction (it uses the pretrained model namely, yolov8n.pt)

# Load the custom YOLOv8 model
model = YOLO("./yolov8m_custom.pt")  

# Load an image
# image_path = 'C:\\Users\\anura\\OneDrive\\Desktop\\python_video_capture\\download.jpeg'
# image = cv2.imread(image_path)

# Perform object detection
# results = model.predict(image)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # '0' is the default camera, change if you have multiple cameras
address = "https://192.168.31.145:8080/video"
cap.open(address)


# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

# Process the video stream frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image from webcam.")
        break

    # Perform object detection
    results = model.predict(source=frame, save=True)

    # Visualize the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates and draw rectangles
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0]
            label = result.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame with bounding boxes
    cv2.imshow('YOLOv8 Real-Time Object Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()