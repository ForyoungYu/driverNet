import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


# 自己设计的模型
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 32, (5, 5), (1, 1), 2),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(2),
                                   nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(2),
                                   nn.Conv2d(32, 64, (5, 5), (1, 1), 2),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(2),
                                   nn.Conv2d(64, 64, (5, 5), (1, 1), 2),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(2),
                                   nn.Conv2d(64, 128, (5, 5), (1, 1), 2),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(2),
                                   nn.Conv2d(128, 128, (5, 5), (1, 1), 2),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(2),
                                   nn.Flatten(), nn.Linear(28800, 128),
                                   nn.Linear(128, 10), nn.LogSoftmax(10))

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    myNet = MyNet().to(device)
    input = torch.ones((64, 3, 480, 480))
    output = myNet(input)
    print(output.shape)
