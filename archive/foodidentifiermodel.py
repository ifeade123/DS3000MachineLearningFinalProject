#Pytorch for machine learning 
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np




#=================================================WARNING!!!!!!!!!!!===================================================
#            Please DO Not run this alot, because you will be making a lot of models and use alot of resources
#=================================================WARNING!!!!!!!!!!!===================================================





#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^DATA PREPROCESSING^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Loading Data^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Loading Model^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#load in the pretrained resenet model
model = models.resnet50(pretrained=True)

# Freeze earlier layers (transfer learning), because it already recognizes stuff
for param in model.parameters():
    param.requires_grad = False #this is what is freezing the model

# Replace final classifier layer with our training
model.fc = nn.Linear(model.fc.in_features, num_classes) #model.fc.in_features gets the features output frim the previous layer, and replaces that layer with our fruit and veggie layer

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = model.to(device)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Training Model^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#train the model
criterion = nn.CrossEntropyLoss() #our neural network loss criteria
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4) #our thing to optimize training, only working on model.fc.paramters because everything else is frozen
epochs = 10 #our rounds of training

#train in different epoch loops
for epoch in range(epochs): #for each epoch
    model.train() #train model, with dropout and batch norm to prevent overfitting
    running_loss = 0.0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) #label the images

        optimizer.zero_grad() #clear the gradients of all optimized tensor
        outputs = model(images) #outputs are what the model has stored at the index images
        loss = criterion(outputs, labels) #The loss criteria is based  on model output and labels
        loss.backward() #calculates discrepancy between predicted and true labels
        optimizer.step() #it is the function that takes the gradients stored in param.grad and applies them to model parameters, using chosen optimization algorithm (ex. SGD, ADAM,)
        running_loss += loss.item() #summing the loss
        
        #the following sums the points of what the model thinks it its most lielly and then it chooses that as its prediction
        _, preds = torch.max(outputs, 1) 
        correct += torch.sum(preds == labels.data)

    accuracy = correct.double() / len(train_data) #calculate accuracy
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f} - Train Accuracy: {accuracy:.4f}") #print to terminal the Epoch, Loss, and Accuracy

model.eval() #parts of the model can act bad during training, so model.eval truns the model into evaluation mode where it can handle stuff better
correct = 0

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Testing Model^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
with torch.no_grad():#dont compute gradients anymore, and this would just be testing the model
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device) #label images

        outputs = model(images)  #output model at index
        _, preds = torch.max(outputs, 1) #prediction
        correct += torch.sum(preds == labels.data)

test_accuracy = correct.double() / len(test_data) #accuracy of the testing
print(f"Test Accuracy: {test_accuracy:.4f}") #print testing accuracy

#^^^^^^^^^^^^^^^^^^^^^^^^Saving Model^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.save(model.state_dict(), "fruitandveggieclassifier.pth") #we save the model as "fruitandveggieclassifier.pth" 