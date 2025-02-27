import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

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
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load dataset using ImageFolder
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

# Data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Function to plot sample images with labels
def plot_sample_images(dataloader, class_names, num_images=12):
    """Plots a batch of sample images with their labels."""
    images, labels = next(iter(dataloader))  # Get a batch
    images, labels = images[:num_images], labels[:num_images]

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))  # 3x4 grid
    fig.suptitle("Sample Images with Labels", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].squeeze(0).numpy()  # Remove channel dimension
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Label: {class_names[labels[i].item()]}")
            ax.axis("off")
    plt.show()

# Plot sample images before training
class_names = full_dataset_train.classes  # Get class names from ImageFolder
plot_sample_images(train_loader, class_names)

# Define the PyTorch CNN Model
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

# Training loop with early stopping
num_epochs = 30
patience = 5
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels.data)
            val_total += inputs.size(0)

    val_loss = val_running_loss / val_total
    val_acc = val_running_corrects.double() / val_total

    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'sd19_model.pth')
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

print("Training complete.")
