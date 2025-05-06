import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import glob


class Sat2RrdDataset(Dataset):
    def __init__(self, dataroot, phase, 
            random_crop=False, random_hflip=False, random_vflip=False, channels=[0,1,2,3]):

        self.phase = phase
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.channels = channels
        self.A_path = os.path.join(dataroot, phase + 'A') #phase
        self.A_path_list = glob.glob(self.A_path+'/*.npy')
        self.dem = np.expand_dims(np.load("./dem_new.npy"), axis=0)[:,450 - 55: 450 + 245, 450 - 82:450 + 193]

    def __len__(self):
        return len(self.A_path_list)

    def _random_crop(self, img_A, img_B, crop_size=(224, 224)):
        # _, height, width
        _, height, width = img_A.shape
        crop_width, crop_height = crop_size

        left = np.random.randint(0, width - crop_width + 1)
        top = np.random.randint(0, height - crop_height + 1)

        imgA = img_A[:, top: top + crop_height, left: left + crop_width]
        imgB = img_B[top: top + crop_height, left: left + crop_width]

        return imgA, imgB

    def _center_crop(self, img_A, img_B, crop_size=(224, 224)):

        _, height, width = img_A.shape
        crop_width, crop_height = crop_size

        left = (width - crop_width) // 2
        top = (height - crop_height) // 2

        imgA = img_A[:, top: top + crop_height, left: left + crop_width]
        imgB = img_B[top: top + crop_height, left: left + crop_width]

        return imgA, imgB

    def _hflip(self, img_A, img_B):
        imgA = np.flip(img_A, axis=2).copy()  
        imgB = np.flip(img_B, axis=1).copy()
        return imgA, imgB

    def _vflip(self, img_A, img_B):
        imgA = np.flip(img_A, axis=1).copy()
        imgB = np.flip(img_B, axis=0).copy()
        return imgA, imgB

    def __getitem__(self, index):

        A_path = self.A_path_list[index]
        B_path = A_path.replace(self.phase+'A', self.phase+'B')

        A = np.load(A_path)
        A = np.concatenate((A, self.dem), axis=0)
        B = np.load(B_path) #/ 100.
        B = np.clip(B, 0, 100)
        B = B/100. #(0~1) # relu
        B = (B*2) -1
        

        if self.phase == 'train':
            if self.random_crop:
                A, B = self._random_crop(A,B)
            else:
                A, B = self._center_crop(A,B)
            if self.random_hflip:
                if np.random.rand() > 0.5:
                    A, B = self._hflip(A, B)
            if self.random_vflip:
                if np.random.rand() > 0.5:
                    A, B = self._vflip(A, B)
        else:
            A, B = self._center_crop(A,B) # center crop

        A = torch.from_numpy(A).float()
        A = A[self.channels, :, :]
        B = torch.from_numpy(B).float().unsqueeze(0)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

if __name__ == "__main__":
    dataset = Sat2RrdDataset(dataroot='/workspace/Sat2RDR_estimation/dataset/train/', phase='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    for idx, data in enumerate(dataloader):
        print(data['A'].shape, data['B'].shape)
