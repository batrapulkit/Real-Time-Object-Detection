import cv2
import numpy as np
import os
import imutils

# Define file paths
weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"
labelsPath = "coco.names"

# Check if all files exist
for file in [weightsPath, configPath, labelsPath]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"[ERROR] {file} not found. Check the path!")

# Load class labels
with open(labelsPath, "r") as f:
    labels = f.read().strip().split("\n")

# Load YOLO model
print("[INFO] Loading YOLO model...")
try:
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
except cv2.error as e:
    print("[ERROR] Failed to load YOLO model! Ensure 'yolov3.cfg' and 'yolov3.weights' are correct.")
    raise e

# Get layer names
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Initialize video stream (Laptop webcam: 0)
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

# Check if webcam is accessible
if not cap.isOpened():
    raise RuntimeError("[ERROR] Unable to access the webcam! Check if it's connected or being used by another app.")

while True:
    ret, frame = cap.read()

    # Skip iteration if frame is empty
    if not ret or frame is None:
        print("[WARNING] Skipping empty frame...")
        continue

    frame = imutils.resize(frame, width=600)

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass
    layerOutputs = net.forward(ln)

    # Initialize lists for detections
    boxes, confidences, classIDs = [], [], []
    H, W = frame.shape[:2]

    # Process detections
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Calculate bounding box coordinates
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply Non-Maxima Suppression (NMS)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Draw bounding boxes
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = (0, 255, 0)  # Green box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[classIDs[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame
    cv2.imshow("YOLO Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
