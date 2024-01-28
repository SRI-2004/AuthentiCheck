import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16
from PIL import Image

# Constants
BASE_IMAGE_PATH = 'train_1/image.jpeg'
BATCH_SIZE = 32
NUM_CLASSES = 2
NUM_AUGMENTATIONS = 10  # Number of augmentations per input image

# Load the base image
base_img = Image.open(BASE_IMAGE_PATH)

# Configure data augmentation transforms
data_transform = transforms.Compose([
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, base_image, transform=None, num_augmentations=1):
        self.base_image = base_image
        self.transform = transform
        self.num_augmentations = num_augmentations

    def __len__(self):
        return self.num_augmentations

    def __getitem__(self, idx):
        try:
            augmented_img = self.transform(self.base_image)
            return augmented_img
        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

        # Create datasets and dataloaders
train_dataset = CustomDataset(base_image=base_img, transform=data_transform, num_augmentations=NUM_AUGMENTATIONS)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Load the pre-trained VGG16 model
base_model = vgg16(pretrained=True)
# Modify the classifier to match the number of classes in your dataset
base_model.classifier[6] = nn.Linear(in_features=4096, out_features=NUM_CLASSES)

# Define the model
model = nn.Sequential(
    base_model,
    nn.Softmax(dim=1)  # Softmax to get class probabilities
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs in train_loader:
        inputs = inputs.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        # Assuming a single class (binary classification)
        labels = torch.ones(inputs.size(0), dtype=torch.long).to(device)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}")

# Save the trained model
torch.save(model.state_dict(), 'vgg_facenet_model.pth')
