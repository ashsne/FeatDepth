

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


