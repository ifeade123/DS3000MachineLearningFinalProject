import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np




#=================================================WARNING!!!!!!!!!!!===================================================
#            Please DO Not run this alot, because you will be making a lot of models and use alot of resources
#=================================================WARNING!!!!!!!!!!!===================================================





# -------------------------
# 1. DATA PREPROCESSING
# -------------------------

# Transformations for training data before we use the data
train_transforms = transforms.Compose([ #compose puts all the transformations of the data in one object
    transforms.Resize((224, 224)), #resizes our images to 224 by 224 because thats what resnet can use
    transforms.RandomHorizontalFlip(), #changing the orientation of the data so that the model can understand changd data
    transforms.RandomRotation(20),#changing the orientation of the data so that the model can understand changd data
    transforms.ColorJitter(brightness=0.2), #changing the orientation of the data so that the model can understand changd data
    transforms.ToTensor(), #makes the image into a tensor so the machine can understand it
    transforms.Normalize([0.485, 0.456, 0.406], #normalizes data to fit in certain range
                         [0.229, 0.224, 0.225])
])

# Transformations for test data
test_transforms = transforms.Compose([ #doing all the transformations for training data
    transforms.Resize((224, 224)), #resize for resnet
    transforms.ToTensor(), #tensor
    transforms.Normalize([0.485, 0.456, 0.406], #normalize
                         [0.229, 0.224, 0.225])
])

# Load dataset directories
train_dir = "fruits-and-vegetables/versions/2/dataset/train" #training data directory
test_dir = "fruits-and-vegetables/versions/2/dataset/test" #test data directory

#use the paths of the train and test data to get the data
train_data = datasets.ImageFolder(train_dir, transform=train_transforms) #transfomed trained data
test_data = datasets.ImageFolder(test_dir, transform=test_transforms) #transformed test data

#load the training  and test data
train_loader = DataLoader(train_data, batch_size=32, shuffle=True) #batch size is how much data the model is dealing with a a time, data loader allows us to wrap the dataset and load it
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

#the number of class is the length of the number of training data classes
num_classes = len(train_data.classes)

# -------------------------
# 2. MODEL SETUP (ResNet-50)
# -------------------------

# Load pretrained ResNet-50
model = models.resnet50(pretrained=True)

# Freeze earlier layers (transfer learning), because it already recognizes stuff
for param in model.parameters():
    param.requires_grad = False #this is what is freezing the model

# Replace final classifier layer with our training
model.fc = nn.Linear(model.fc.in_features, num_classes) #model.fc.in_features gets the features output frim the previous layer, and replaces that layer with our fruit and veggie layer

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = model.to(device)

# -------------------------
# 3. TRAINING SETUP
# -------------------------

criterion = nn.CrossEntropyLoss() #our neural network loss criteria
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4) #our thing to optimize training, only working on model.fc.paramters because everything else is frozen
epochs = 10 #our rounds of training

# -------------------------
# 4. TRAINING LOOP
# -------------------------

for epoch in range(epochs): #for each epoch
    model.train() #train model, with dropout and batch norm to prevent overfitting
    running_loss = 0.0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) #assign it

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

torch.save(model.state_dict(), "fruitandveggieclassifier.pth") #we save the model as "fruitandveggieclassifier.pth" 