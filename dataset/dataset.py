import os
import matplotlib.pyplot as plt 
import numpy as np

from PIL import Image

# torch libraries
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

## Functions and classes
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class MonoDataset(Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        is_train
    """

    def __init__(self,
                config,
                filenames,
                is_train=True):
        self.data_path = config.data_path
        self.height = config.height
        self.width = config.width
        self.transforms = config.transforms
        self.filenames = filenames
        self.is_train = is_train
        super().__init__()

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        folder, image_idx, side = self.filenames[idx].split(" ")
        image_path = self.getImagePath(folder, image_idx, side)
        image = pil_loader(image_path)
        image = self.preprocess(image)
        return image

    def preprocess(self, image):
        if self.transforms is not None:
            transform = self.transforms["train" if self.is_train else "val"]
            image = transform(image)
        #print(image_path)
        return image
    
    def getImagePath(self, folder, image_idx, side):
        image_name = str("{:010d}".format(int(image_idx))) + ".jpg"
        image_path = os.path.join(self.data_path, folder, "image_03/data", image_name)
        return image_path

def readlines(filename):
    """Read all the lines from a file and return a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def get_dataset(config, is_train):
    datafile_path = os.path.join(os.path.dirname(__file__), "splits", "{}_files.txt")
    split_filenames = (datafile_path.format("train" if is_train else "val"))
    filenames = readlines(split_filenames)
    dataset = MonoDataset(config, filenames, is_train)
    return dataset

class Config():
    def __init__(self):
        self.data_path = "/media/ash/ashish/ashish/data/kitti_depth/kitti_data"
        self.height = 320
        self.width = 1024
        self.batch_size = 4
        # Data augmentation and normalization for training
        # Just normalization for validation
        self.transforms = {
            'train': transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor()
            ]),
        }


def show_image(img):
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    config = Config()
    train_dataset = get_dataset(config, is_train=True)
    image = train_dataset.__getitem__(3232)
    #show_image(image)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = iter(train_loader)
    images  =train_iter.next()
    print('images shape on batch size = {}'.format(images.size()))
    show_image(transforms.ToPILImage()(image))

    val_dataset = get_dataset(config, is_train=False)
