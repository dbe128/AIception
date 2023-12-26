import torch
import torchvision.transforms as transforms
from Model import CatDogClassifier
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# Create an instance of the CNN
model = CatDogClassifier()

# Load the saved model from file
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Set the model in evaluation mode

batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image tensors
])

test_dataset = ImageFolder("/Users/dbe128/IdeaProjects/AIception/data/test", transform=transform)


test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Variables to track accuracy
correct_predictions = 0
total_predictions = 0

# Disable gradient calculation for evaluation
with torch.no_grad():
    for images, labels in test_loader:
        # Move images and labels to the device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass to get predictions
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Update the accuracy counts
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

# Calculate the accuracy
accuracy = correct_predictions / total_predictions
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
