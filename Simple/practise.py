import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from PIL import Image

train_dataset = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor, download=True)
test_dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=8)


class TrainNet(nn.Module):
    def __init__(self):
        super(TrainNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), (1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (1, 1)),

        )

for data in train_loader:
    img, target = data
    print(img)
    print(target)