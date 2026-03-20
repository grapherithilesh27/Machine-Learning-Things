# Import necessary libraries
import cv2
import numpy as np
import torch

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define function to detect vehicles
def detect_vehicles(image):
    # Get image dimensions
    (H, W) = image.shape[:2]

    # Convert image to blob and pass through YOLOv3 model
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists to store detected vehicles and their coordinates
    vehicles = []
    coordinates = []

    # Loop through each output
    for output in outputs:
        # Loop through each detection
        for detection in output:
            # Get scores and class ID
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak predictions
            if confidence > 0.5 and classes[classID] in ["car", "truck", "bus"]:
                # Get coordinates and draw bounding box
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

                # Add vehicle and coordinates to lists
                vehicles.append(classes[classID])
                coordinates.append((x, y, x + width, y + height))

    # Return image with detected vehicles and their coordinates
    return image, vehicles, coordinates

# Test the function
image = cv2.imread("test_image.jpg")
image, vehicles, coordinates = detect_vehicles(image)
print(vehicles)
print(coordinates)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()