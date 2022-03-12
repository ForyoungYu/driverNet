import torch.nn as nn
import torch
from PIL import Image
import torchvision


test_img = "archive/imgs/test/img_79637.jpg"
img = Image.open(test_img)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((240, 240)),
    torchvision.transforms.ToTensor()
])

img = transform(img)

model = torch.load("saved_model/model_epoch1.pth")
img = torch.reshape(img, (1, 3, 240, 240))
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))
