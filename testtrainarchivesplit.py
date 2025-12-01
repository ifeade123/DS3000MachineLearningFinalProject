

'''
below for loadiung model in diufferent cide
import torch
from torchvision import models
import torch.nn as nn

num_classes =  your_number_of_classes_here

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load("fruit_classifier.pth", map_location="cpu"))
model.eval()

print("Model loaded!")

below for putting an image in
from PIL import Image
from torchvision import transforms
import torch

# same normalization used in training!
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img = Image.open("test_image.jpg")
img = transform(img).unsqueeze(0)   # add batch dimension

with torch.no_grad():
    output = model(img)
    _, pred = torch.max(output, 1)

print("Prediction:", pred.item())

'''