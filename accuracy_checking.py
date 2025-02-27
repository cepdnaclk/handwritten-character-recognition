import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Set device for GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data directory
data_dir = 'by_merge'

# Define transforms for training (with data augmentation) and validation
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),  # Converts image to [0, 1] float tensor
])

transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create full datasets (using ImageFolder, which assumes subdirectories per class)
full_dataset_train = datasets.ImageFolder(root=data_dir, transform=transform_train)
full_dataset_val = datasets.ImageFolder(root=data_dir, transform=transform_val)

# Create a split (80% training, 20% validation)
num_samples = len(full_dataset_train)
indices = list(range(num_samples))
np.random.shuffle(indices)
split = int(np.floor(0.2 * num_samples))
train_idx, val_idx = indices[split:], indices[:split]

train_dataset = Subset(full_dataset_train, train_idx)
val_dataset = Subset(full_dataset_val, val_idx)

# DataLoader for training and validation datasets
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Select 100 random samples from the training set
random.seed(42)  # Set seed for reproducibility
random_samples = random.sample(range(len(train_dataset)), 100)

# Create a subset of 100 random images from the training set
subset_train_dataset = Subset(train_dataset, random_samples)

# DataLoader for this subset
subset_train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Define the model
class SD19Model(nn.Module):
    def __init__(self, num_classes=47):
        super(SD19Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),  # input channel=1 for grayscale
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self._init_linear()
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _init_linear(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 128)
            features = self.features(dummy)
            self.flattened_size = features.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

# Initialize the model, loss function, and optimizer
model = SD19Model(num_classes=47).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the best saved model (make sure the file exists)
model.load_state_dict(torch.load('sd19_model.pth', map_location=torch.device('cpu')))
model.eval()

# Evaluate on the 100 random images
subset_running_loss = 0.0
subset_running_corrects = 0
subset_total = 0

# Store images, predictions, and actual labels for visualization
images = []
predictions = []
actual_labels = []

with torch.no_grad():
    for inputs, labels in subset_train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        subset_running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        subset_running_corrects += torch.sum(preds == labels.data)
        subset_total += inputs.size(0)

        # Store images, predictions, and actual labels
        images.extend(inputs.cpu())
        predictions.extend(preds.cpu())
        actual_labels.extend(labels.cpu())

# Calculate final loss and accuracy
subset_loss = subset_running_loss / subset_total
subset_acc = subset_running_corrects.double() / subset_total

print(f"Subset Loss: {subset_loss:.4f}, Subset Accuracy: {subset_acc:.4f}")

# Sort images, predictions, and actual labels for orderly display
sorted_data = sorted(zip(images, predictions, actual_labels), key=lambda x: x[2].item())
images, predictions, actual_labels = zip(*sorted_data)

# Visualize the images with predictions and actual labels
fig, axes = plt.subplots(10, 10, figsize=(15, 15))
axes = axes.flatten()

for img, pred, actual, ax in zip(images, predictions, actual_labels, axes):
    img = img.squeeze().numpy()
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Pred: {pred.item()}, Actual: {actual.item()}")
    ax.axis('off')

plt.tight_layout()
plt.show()