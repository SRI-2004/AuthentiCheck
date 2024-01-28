import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import vgg16
from torch import nn
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
        input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

        # Run the object detection model
        with torch.no_grad():
            prediction = self.object_detection_model(input_batch)

        # Filter predictions based on score threshold
        filtered_indices = [i for i, score in enumerate(prediction[0]['scores']) if score > score_threshold]

        num_humans = len(filtered_indices)
        print(num_humans)
        if num_humans > 1:
            return True
        else:
            return False


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
        input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

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
