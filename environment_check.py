import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

import torchvision.transforms as transforms
class EnvironmentDetection:
    def __init__(self):
        # Load a pre-trained Faster R-CNN model for object detection (ResNet-50)
        self.object_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.object_detection_model.eval()

        # Convert the frame to a PIL Image

    def verify(self,frame):
        # Convert the PIL Image to a PyTorch tensor
        frame_pil = Image.fromarray(frame)

        # Convert the frame to a PyTorch tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        input_tensor = transform(frame_pil)
        input_batch = input_tensor.unsqueeze(0)
        # Add a batch dimension
        # Run the object detection model
        with torch.no_grad():
            prediction = self.object_detection_model(input_batch)

        # Print the list and number of objects detected
        if 'boxes' in prediction[0]:
            num_objects = len(prediction[0]['boxes'])
            print(f"Number of Objects Detected: {num_objects}")
            print("List of Objects:")
            for i in range(num_objects):
                box = prediction[0]['boxes'][i]
                print(f"Object {i + 1}: {box}")
        else:
            print("No objects detected.")
