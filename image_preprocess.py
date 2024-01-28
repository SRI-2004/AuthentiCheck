import torch
from torchvision.models import vgg16
from torchvision import transforms
from PIL import Image
import os
from torch import nn
# Constants
BASE_IMAGE_PATH = 'train_1/face.jpg'
NUM_CLASSES = 2

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
img = Image.open(BASE_IMAGE_PATH)
img = transform(img)
img = img.unsqueeze(0)  # Add a batch dimension

# Create a VGG16 model
base_model = vgg16(pretrained=True)
# Modify the classifier to match the number of classes in your dataset
base_model.classifier[6] = nn.Linear(in_features=4096, out_features=NUM_CLASSES)

# Define the model
model = nn.Sequential(
    base_model,
    nn.Softmax(dim=1)  # Softmax to get class probabilities
)


# Load the pre-trained weights
model.load_state_dict(torch.load('vgg_facenet_model.pth'))

# Set the model to evaluation mode
model.eval()

# Perform inference
# Perform inference
with torch.no_grad():
    output = model(img)

# Get the predicted class
print("Class Probabilities:", output)
predicted_class = torch.argmax(output).item()

print(f"Predicted Class: {predicted_class}")
