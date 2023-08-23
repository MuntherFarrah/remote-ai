import cv2
import numpy as np

# Load the YOLO network
net = cv2.dnn.readNet("../data/yolov3.weights", "../data/yolov3.cfg")

# Load the COCO class names
with open("../data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the output layer names
layer_names = net.getLayerNames()
output_layer_indices = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in output_layer_indices]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    height, width, _ = img.shape

    # Convert image to blob and perform forward pass
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indexes.flatten():  # Flattening the indexes array
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Show the image with OpenCV
    cv2.imshow("Image", img)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
