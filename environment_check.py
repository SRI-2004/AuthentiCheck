import torch
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

class EnvironmentDetection:
    def __init__(self):
        # Load a pre-trained YOLOv5 model
        self.object_detection_model = attempt_load('yolov5s.pt',device="cuda")
        self.object_detection_model.eval()

    def verify(self, frame):
        # Convert the PIL Image to a PyTorch tensor
        frame_pil = Image.fromarray(frame)

        # Convert the frame to a PyTorch tensor
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(frame_pil)
        input_batch = input_tensor.unsqueeze(0).to("cuda")

        # Run the object detection model
        model = self.object_detection_model
        model.to("cuda")
        with torch.no_grad():
            prediction = model(input_batch)

        # Post-process the prediction to get class labels and boxes
        # Assuming class 0 corresponds to person, and class 67 corresponds to a cell phone in COCO dataset
        # Post-process the prediction to get class labels and boxes
        # Assuming class 0 corresponds to person, and class 67 corresponds to a cell phone in COCO dataset
        class_labels = prediction[0][:, -1]
        boxes = prediction[0][:, :-1]

        # Apply non-maximum suppression
        detections = non_max_suppression(boxes, conf_thres=0.5, iou_thres=0.5)

        # Assuming class 0 corresponds to person, and class 67 corresponds to a cell phone in COCO dataset
        person_count = 0
        phone_detected = False

        for detection in detections:
            if detection is not None and len(detection) > 0:
                class_ids = detection[:, -1]
                if 0 in class_ids:
                    person_count += 1
                if 73 or 77 in class_ids:
                    phone_detected = True

        if person_count > 1 and phone_detected:
            return 1  # More than one person and a phone detected
        elif person_count > 1:
            return 2  # More than one person detected
        elif phone_detected:
            return 3  # One person and a phone detected
        else:
            return 4  # Other objects detected (can be adjusted based on actual class labels)



# env_detection = EnvironmentDetection()

# # Open a connection to the camera (camera index 0 by default)
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()
#
# while True:
#     # Capture a frame from the camera
#     ret, frame = cap.read()
#
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break
#
#     # Perform object detection and get the result
#     result = env_detection.verify(frame)
#
#     # Display the result
#     if result == 1:
#         print("More than one person and a phone detected.")
#     elif result == 2:
#         print("More than one person detected.")
#     elif result == 3:
#         print("One person and a phone detected.")
#     # elif result == 4:
#     #     print("One person detected")
#     # else:
#     #     print("Unknown result.")
#
#     # Display the frame with detected objects
#     cv2.imshow("Frame", frame)
#
#     # Break the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()
