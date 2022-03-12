import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os
import pandas as pd
from PIL import Image
import re


class DriverDataset(Dataset):
    def __init__(self,
                 root_dir,
                 train=True,
                 csv_file: str = None,
                 transform=None,
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.root_dir = root_dir
        self.csv_file = csv_file
        if train:
            self.train_or_test = 'train'
        else:
            self.train_or_test = 'test'
        if self.csv_file:
            self.csv_data = pd.read_csv(csv_file)

    def __getitem__(self, item):
        img_name, label = self.csv_data['img'][item], self.csv_data[
            'classname'][item]
        img_path = os.path.join(self.root_dir, self.train_or_test, label,
                                img_name)
        img = Image.open(img_path)

        label = re.findall("\d$", label)
        # label = F.one_hot(int(label[0]), num_classes=9)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, int(label[0])

    def __len__(self):
        return len(self.csv_data)


if __name__ == '__main__':
    root_dir = "archive/imgs"
    csv_file = "archive/driver_imgs_list.csv"

    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
    ])
    train_dataset = DriverDataset(root_dir,
                                  train=True,
                                  csv_file=csv_file,
                                  transform=transform)
    # test_dataset = DriverDataset(train=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=8)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    print(len(train_dataset))
