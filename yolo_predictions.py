import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader


class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        # load YAML
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # load YOLO model
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
        # step-1 convert image into square image (array)
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image
        # step-2: get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()  # detection or prediction from YOLO

        # Non Maximum Suppression
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # width and height of the image (input_image)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # confidence of detection an object
            if confidence > 0.4:
                class_score = row[5:].max()  # maximum probability from 20 objects
                class_id = row[5:].argmax()  # get the index position at which max probability occurs

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # construct bounding box from four values
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])

                    # append values into the list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # clean
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # NMS (Non-Maximum Suppression)
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()

        # Draw the Bounding Boxes
        for ind in index:
            # extract bounding box
            x, y, w, h = boxes_np[ind]
            bb_conf = int(confidences_np[ind] * 100)
            classes_id = classes[ind]
            class_name = self.labels[classes_id]
            colors = self.get_color(classes_id)

            text = f'{class_name}: {bb_conf}%'

            # Increase bounding box thickness to 3
            cv2.rectangle(image, (x, y), (x + w, y + h), colors, 3)

            # Draw a filled rectangle behind the text for better readability
            cv2.rectangle(image, (x, y - 30), (x + w, y), colors, -1)

            # Increase text size and thickness for better visibility
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)

        return image

    def get_color(self, ID):
        # Return color based on class ID
        return self.colors.get(ID, (255, 255, 255))  # Default to white if ID is not found

# Example usage:
# model = YOLO_Pred('yolov5.onnx', 'data.yaml')
# img = cv2.imread('test_image.jpg')
# result = model.predictions(img)
# cv2.imshow('YOLO Detection', result)
# cv2.waitKey(0)