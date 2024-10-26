import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        # Load YAML
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # Load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Define the colors
        self.colors = {
            0: (213, 41, 73),   # #d52941 in BGR
            1: (215, 133, 33),  # #d78521 in BGR
            2: (32, 32, 32)     # #202020 in BGR
        }

    def predictions(self, image):
        row, col, d = image.shape
        # Step-1: Convert image into square image (array)
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # Step-2: Get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()  # Detection or prediction from YOLO

        # Non-Maximum Suppression
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # Width and height of the image (input_image)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # Confidence of detecting an object
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # Construct bounding box
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])

                    # Append values to lists
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # Convert to lists for NMS
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # Check if there are any boxes to apply NMS
        if boxes_np and confidences_np:
            index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

            # Ensure that NMS found any valid boxes
            if len(index) > 0:
                index = index.flatten()
                for ind in index:
                    x, y, w, h = boxes_np[ind]
                    bb_conf = int(confidences_np[ind] * 100)
                    classes_id = classes[ind]
                    class_name = self.labels[classes_id]
                    colors = self.get_color(classes_id)

                    text = f'{class_name}: {bb_conf}%'

                    # Draw bounding box with text
                    cv2.rectangle(image, (x, y), (x + w, y + h), colors, 3)
                    cv2.rectangle(image, (x, y - 30), (x + w, y), colors, -1)
                    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)
            else:
                # No valid boxes after NMS
                cv2.putText(image, "No objects detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # No boxes detected
            cv2.putText(image, "No objects detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return image

    def get_color(self, ID):
        # Return color based on class ID
        return self.colors.get(ID, (255, 255, 255))  # Default to white if ID is not found
