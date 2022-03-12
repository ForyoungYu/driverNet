import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from driverdataset import DriverDataset
from torch.utils.data import DataLoader
import NetModule
from torch.utils.tensorboard.writer import SummaryWriter

if __name__ == '__main__':
    root_dir = "dataset/archive/imgs"
    csv_file = "dataset/archive/driver_imgs_list.csv"
    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
    ])

    train_dataset = DriverDataset(root_dir,
                                  train=True,
                                  csv_file=csv_file,
                                  transform=transform)
    # test_dataset = DriverDataset(train=False)

    # train_data_size = len(train_dataset)
    # test_data_size = len(test_dataset)
    # print("训练集长度：{}".format(train_data_size))
    # print("测试集长度：{}".format(test_data_size))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4,
                                  shuffle=True,
                                  num_workers=8)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    # 定义GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建训练模型
    # model = NetModule.DriverNet().to(device)
    vgg11 = torchvision.models.vgg11()

    # 修改模型
    vgg11.features[0] = nn.Conv2d(3, 64, (3, 3), (1, 1), 2)
    vgg11.features[3] = nn.Conv2d(64, 128, (3, 3), (1, 1), 2)
    vgg11.features[6] = nn.Conv2d(128, 256, (3, 3), (1, 1), 2)
    vgg11.features[8] = nn.Conv2d(256, 256, (3, 3), (1, 1), 2)
    vgg11.features[11] = nn.Conv2d(256, 512, (3, 3), (1, 1), 2)
    vgg11.features[13] = nn.Conv2d(512, 512, (3, 3), (1, 1), 2)
    vgg11.features[16] = nn.Conv2d(512, 512, (3, 3), (1, 1), 2)
    vgg11.features[18] = nn.Conv2d(512, 512, (3, 3), (1, 1), 2)
    vgg11.classifier[3] = nn.Linear(4096, 2048)
    vgg11.classifier[6] = nn.Linear(2048, 1024)
    vgg11.add_module(
        "output",
        nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True),
                      nn.Dropout(p=0.5, inplace=False), nn.Linear(512, 10)))

    vgg11 = vgg11.to(device)
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 优化器
    learing_rate = 1e-2
    optimizer = torch.optim.Adam(vgg11.parameters(), lr=learing_rate)

    total_train_step = 0
    total_test_step = 0
    epoch = 10

    # writer = SummaryWriter("./logs")
    for i in range(epoch):
        print("============ EPOCH: {} ===========".format(i + 1))
        for data in train_dataloader:
            imgs, targets = data

            # 将target表示成one_hot表示法
            target = F.one_hot(targets)

            torch.cuda.empty_cache()
            # 将图像和标签传入gpu
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = vgg11(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            # if total_train_step % 100 == 0:
            print("训练次数：{}， loss：{}".format(total_train_step, loss.item()))
            # writer.add_scalar("train_loss", loss.item(), total_train_step)

        torch.save(vgg11, "saved_model/model_epoch{}.pth".format(i + 1))
        print("模型已保存。")

    # writer.close()
