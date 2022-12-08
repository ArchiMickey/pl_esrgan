import pytorch_lightning as pl
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from loguru import logger
from tqdm import tqdm

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    tensors = tensors.permute(1, 2, 0) * (std * 255) + (mean * 255)
    return torch.clamp(tensors, 0, 255).permute(2, 0, 1)


class SRDataset(Dataset):
    def __init__(self, pathlist, hr_shape, mode='train') -> None:
        super().__init__()
        self.hr_shape = hr_shape
        hr_height, hr_width = self.hr_shape
        self.mode = mode
        if self.mode == 'train':
            self.hr_transform = A.Compose([
                A.RandomCrop(hr_height, hr_width),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.Resize(hr_height, hr_width, interpolation=Image.BICUBIC),
                A.Normalize(),
            ])
            self.lr_transform = A.Resize(hr_height // 4, hr_width // 4, interpolation=Image.BICUBIC)
        elif self.mode == 'val':
            self.hr_shape = (512, 512)
            hr_height, hr_width = self.hr_shape
            self.hr_transform = A.Compose([
                A.CenterCrop(512, 512),
                A.Normalize(),
            ])
            self.lr_transform = A.Resize(512 // 4, 512 // 4, interpolation=Image.BICUBIC)
        self.final_transform = ToTensorV2()
        self.files = []
        for path in pathlist:
            valid_filelist = []
            logger.info(f"Loading images from {path}")
            for path in tqdm(glob.glob(path + "/*.*")):
                img = Image.open(path)
                h, w = img.size
                if h >= hr_height and w >= hr_width:
                    valid_filelist.append(path)
            self.files += valid_filelist

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hr = self.hr_transform(image=img)['image']
        img_lr = self.lr_transform(image=img_hr)['image']
        img_hr, img_lr = self.final_transform(image=img_hr)['image'], self.final_transform(image=img_lr)['image']
        return {"lr": img_lr, "hr": img_hr}
    
class SRDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, pathlist_dict, hr_height, hr_width) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pathlist_dict = pathlist_dict
        self.hr_shape = (hr_height, hr_width)
    
    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_ds = SRDataset(self.pathlist_dict["train"], self.hr_shape)
            self.val_ds = SRDataset(self.pathlist_dict["val"], self.hr_shape, mode='val')
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)