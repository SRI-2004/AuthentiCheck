import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import vgg16
from torch import nn
from yolov5.models.experimental import attempt_load
import  cv2
from yolov5.utils.general import non_max_suppression

class FaceVerification:
    def __init__(self):
        # Load a pre-trained Faster R-CNN model for object detection
        self.object_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.object_detection_model.eval()

        # Transformation for the input image
        self.transform = transforms.Compose([transforms.ToTensor()])

    def human_detection(self, frame, score_threshold=0.8):
        # Convert the frame to a PyTorch tensor
        input_tensor = self.transform(frame)
        input_batch = input_tensor.unsqueeze(0)
        model = self.object_detection_model
        # Run the object detection model
        with torch.no_grad():
            prediction = model(input_batch)

        # Filter predictions based on score threshold
        filtered_indices = [i for i, score in enumerate(prediction[0]['scores']) if score > score_threshold]
        filtered_indices = [i for i in filtered_indices if prediction[0]['labels'][i] == 1]
        print(prediction[0]["labels"])
        num_humans = len(filtered_indices)
        print(num_humans)
        if num_humans > 1:
            return True
        elif num_humans == 0:
            print("Error identifying, please try again")
            return
        else:
            return False

    # More than one person detected

    def face_detection(self, frame):
        # Convert the NumPy array to a PIL Image
        frame_pil = Image.fromarray(frame)

        # Convert the frame to a PyTorch tensor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(frame_pil)
        input_batch = input_tensor.unsqueeze(0)
        # Add a batch dimension
        # Load the pre-trained VGG16 model
        base_model = vgg16(pretrained=True)
        # Modify the classifier to match the number of classes in your dataset
        base_model.classifier[6] = nn.Linear(in_features=4096, out_features=2)

        # Define the model
        model = nn.Sequential(
            base_model,
            nn.Softmax(dim=1)  # Softmax to get class probabilities
        )

        # Load the pre-trained weights
        model.load_state_dict(torch.load('vgg_facenet_model.pth'))
        model.eval()
        # Run the face recognition model
        with torch.no_grad():
            output = model(input_batch)

        # Get the predicted class
        print("Class Probabilities:", output)
        predicted_class = torch.argmax(output, dim=1).item()

        return predicted_class

    def verify(self, frame):
        # Check if more than one human is present
        if self.human_detection(frame):
            print("Error: More than one human detected.")
            return 2

        # Check if at least one face is detected
        if not self.face_detection(frame):
            print("Error: No face detected.")
            return 0

        # For simplicity, this example returns 1 (verification successful)
        print("Face Verified")
        return 1


face_verification = FaceVerification()

# Open a connection to the camera (camera index 0 by default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break
    cv2.imshow("Frame", frame)
    # Perform verification on the frame
    result = face_verification.verify(frame)

    # Display the result
    # if result == 1:
    #     print("Verification successful")
    # elif result == 2:
    #     print("More than one human detected")
    # elif result == 0:
    #     print("No face detected")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
