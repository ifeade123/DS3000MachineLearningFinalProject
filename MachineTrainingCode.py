import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np

# -------------------------
# 1. DATA PREPROCESSING
# -------------------------

# Transformations for training data
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Transformations for test data
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset directories
train_dir = "/content/drive/MyDrive/archive"
test_dir = "/content/drive/MyDrive/archive"

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

num_classes = len(train_data.classes)

# -------------------------
# 2. MODEL SETUP (ResNet-50)
# -------------------------

# Load pretrained ResNet-50
model = models.resnet50(pretrained=True)

# Freeze earlier layers (transfer learning)
for param in model.parameters():
    param.requires_grad = False

# Replace final classifier layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -------------------------
# 3. TRAINING SETUP
# -------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
epochs = 10

# -------------------------
# 4. TRAINING LOOP
# -------------------------

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)

    accuracy = correct.double() / len(train_data)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f} - Train Accuracy: {accuracy:.4f}")
# -------------------------
# 5. EVALUATION
# -------------------------

model.eval()
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)

test_accuracy = correct.double() / len(test_data)
print(f"Test Accuracy: {test_accuracy:.4f}")