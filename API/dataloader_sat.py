import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import os.path as osp
import cv2
import torch
import glob
import torch.nn.functional as F
from torch.utils.data import Dataset
import tqdm

class SatelliteBenchDataset(Dataset):
    def __init__(self, data_root, data_split):
        super().__init__()
        data_root = '/workspace/SSD_4T_b/sat_dataset'
        self.data_root = data_root
        self.data_split = data_split
        self.data_path = sorted(glob.glob(osp.join(data_root, self.data_split, '*', '*')))
        if data_split == 'train':
            self.data_path = self.data_path * 20
        self.data_limit = len(self.data_path)
        self.data_name = "satellite"
        self.mean = 0
        self.std = 1
        self.days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        #self.dem = np.expand_dims(np.load("./dem_900.npy"), axis=0)
        self.dem = np.expand_dims(np.load("/workspace/SSD_4T_b/Implicit-Stacked-Autoregressive-Model-for-Video-Prediction/dem_new.npy"), axis=0)

    def __len__(self):
        return len(self.data_path)

    def __stack_data__(self, path):
        imgs = []
        for i in path:
            im = np.load(i)
            stacked_array = np.concatenate((im, self.dem), axis=0)
            imgs.append(stacked_array)
        imgs = np.stack(imgs, axis=0)
        imgs = torch.from_numpy(imgs).float()
        return imgs

    def __random_crop__(self, inputs, gt, crop_size=768):
        _, _, h, w = inputs.shape
        
        if crop_size > h or crop_size > w:
            raise ValueError("Crop size must be smaller than the image dimensions")

        top = torch.randint(0, h - crop_size + 1, (1,)).item()
        left = torch.randint(0, w - crop_size + 1, (1,)).item()

        inputs_cropped = inputs[:, :, top:top + crop_size, left:left + crop_size]
        gt_cropped = gt[:, :, top:top + crop_size, left:left + crop_size]

        return inputs_cropped, gt_cropped

    def __getitem__(self, index):
        videos = glob.glob(self.data_path[index] + '/*')
        videos = sorted(videos)
        start_index = np.random.randint(len(videos)-12)
        time = videos[start_index:start_index+6][-1]
        time = osp.basename(time)[:-4]
        month = int(time[4:6])
        day = int(time[6:8])
        day = sum(self.days_in_month[:month-1]) + day
        hour = int(time[8:-2])
        inputs = videos[start_index:start_index+6]
        outputs = videos[start_index+6: start_index+12]
        inputs = self.__stack_data__(inputs)
        outputs = self.__stack_data__(outputs)
        inputs, outputs = self.__random_crop__(inputs, outputs)

        return inputs, outputs[:,:3,:,:], day, hour

if __name__ == "__main__":
    dataset = SatelliteBenchDataset(data_root='s', data_split='train')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=16)

    for idx, i in enumerate(train_loader):
        print(i[0].min(), i[0].max())

