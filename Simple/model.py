import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), (1, 1), 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 5), (1, 1), 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 测试网络
if __name__ == '__main__':
    myNet = MyNet()
    input = torch.ones((64, 3, 32, 32))
    output = myNet(input)
    print(output.shape)
