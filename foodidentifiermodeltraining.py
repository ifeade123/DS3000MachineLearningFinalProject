import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
from PIL import Image
import os

#======================================================================
# Create a tuple of names for the indexes of prediction so we get a useful output
#======================================================================
train_dir = "fruits-and-vegetables/versions/2/dataset/train"
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
#print(f"Number of classes: {len(class_names)}") #for troubleshooting only
#print(f"Classes: {class_names}") #for troubleshooting only

#Bring In our Model
model_path = "fruitandveggieclassifier.pth"

#Load In The Rest of Pretrained ResNet
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

state_dict = torch.load(model_path, map_location="cpu") #load model
num_classes = state_dict['fc.weight'].shape[0] #the number of classes in the model
#below loads in the neural network
model.fc = nn.Linear(model.fc.in_features, num_classes) #the fully connected layer
model.load_state_dict(state_dict) #load it!
model.eval()
print("Model loaded!")

#Image Preprocessing for resnet
transform = transforms.Compose([
    transforms.Resize((224, 224)), #resize photo to 224 by 224 because thats what resnet will take
    transforms.ToTensor(), #make it a tensor
    transforms.Normalize([0.485, 0.456, 0.406], #nromalize he data
                         [0.229, 0.224, 0.225])
])

#open the image
img = Image.open("test_image.jpg")
img = transform(img).unsqueeze(0) #torch needs a batch dimension input for the model to understand your image, so the unsqueezing does that, and the transform applies the above transformations to our image

#ACTUAL MODEL PREDICTION BELOW
with torch.no_grad(): #so basically telling the model to not compute gradients because we arent training the model anymore
    output = model(img)#sends model to our neural network for prediction
    _, pred = torch.max(output, 1) #finds the index of the highest score, which means the model thinks that it is most likely whatever it is predicting

#print(f"Prediction index: {pred.item()}") #for testing only
print(f"Prediction name: {class_names[pred.item()]}")



