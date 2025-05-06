import logging
import os
import pprint
import random
from tqdm import tqdm
import argparse

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader


from models.pix2pixhd  import Generator
from dataloader import Sat2RrdDataset 

from metrics import cal_csi
from utils import save_images


"""
Clean codes are used relative path!! (I am used to using absolute paths to better understand the folder structure.) 
"""

parser = argparse.ArgumentParser(description='PyTorch sat2rdr Training')
parser.add_argument('--num_workers', default=4, type=float, help='cpu number')

# python3 train.py --random_crop True --random_hflip True --random_vflip True
parser.add_argument('--gpu_id', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--dataroot', default='./dataset/val', type=str, help='data path')
parser.add_argument('--channels', default='0 1 2 3', type=str, help='List of channel indices to use.')

parser.add_argument('--in_ch', default=4, type=int, help='input channels')
parser.add_argument('--out_ch', default=1, type=int, help='output channels')

parser.add_argument('--out_dir', default='./results_30', type=str, help='result path')
parser.add_argument('--exp_name', default='', help='identifier for experiment')

parser.add_argument('--mid_type', default='van', type=str, help='[resnet, van, convnext]')
parser.add_argument('--act_type', default='silu', type=str, help='relu, silu, gelu')
# /workspace/Sat2RDR_estimation/results_van_silu/190.pth
parser.add_argument('--ckpt_path', default='./results_van_silu_trainf/200.pth', help='identifier for experiment')
parser.add_argument('--save_img_path', default='./saved_img/', help='identifier for experiment')
args = parser.parse_args()

cudnn.benchmark = True

args.channels = [int(i) for i in args.channels.split(' ')]

val_dataset = Sat2RrdDataset(dataroot=args.dataroot, phase='val', channels=args.channels)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers)

model_G = Generator(input_ch=args.in_ch, act_type=args.act_type, mid_type=args.mid_type)

model_G.to('cuda:{}'.format(args.gpu_id))
model_G = torch.nn.DataParallel(model_G)
checkpoint = torch.load(args.ckpt_path)
model_G.load_state_dict(checkpoint)

val_total_iters = len(val_dataloader)

print(val_total_iters)
def val():

    csi_1mm, pod_1mm, far_1mm = 0, 0, 0
    csi_4mm, pod_4mm, far_4mm = 0, 0, 0
    csi_8mm, pod_8mm, far_8mm = 0, 0, 0

    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False)
    
    total_squared_error = 0
    total_samples = 0

    for idx, batchs in pbar:
        model_G.eval()

        with torch.no_grad():
            inputs, targets = batchs['A'].to('cuda:{}'.format(args.gpu_id)), batchs['B'].to('cuda:{}'.format(args.gpu_id))
            A_path, B_path =  batchs['A_paths'], batchs['B_paths']
            pred = model_G(inputs)

            pred = ((pred+1)/2) * 100.
            targets = ((targets+1)/2) * 100.

            csi, pod, far  = cal_csi(pred, targets, threshold=0.1)
            csi_1mm += csi
            pod_1mm += pod
            far_1mm += far

            csi, pod, far  = cal_csi(pred, targets, threshold=4.0)
            csi_4mm += csi
            pod_4mm += pod
            far_4mm += far

            csi, pod, far = cal_csi(pred, targets, threshold=8.0)
            csi_8mm += csi
            pod_8mm += pod
            far_8mm += far

            total_squared_error += torch.sum((pred - targets) ** 2).item()
            total_samples += targets.numel()

        pbar.set_postfix({
            "CSI 1mm": f"{csi_1mm / (idx + 1):.4f}", 
            "CSI 4mm": f"{csi_4mm / (idx + 1):.4f}",
            "CSI 8mm": f"{csi_8mm / (idx + 1):.4f}"
        })
        out_dir = args.save_img_path + args.exp_name
        #save_images(out_dir, idx, inputs[-1, 0, :,:], targets[-1,0,:,:], pred[-1,0,:,:], A_path[-1], B_path[-1])
    n_batches = len(val_dataloader)

    csi_1mm /= n_batches
    pod_1mm /= n_batches
    far_1mm /= n_batches

    csi_4mm /= n_batches
    pod_4mm /= n_batches
    far_4mm /= n_batches

    csi_8mm /= n_batches
    pod_8mm /= n_batches
    far_8mm /= n_batches

    rmse = (total_squared_error / total_samples) ** 0.5

    print(f"1mm - CSI: {csi_1mm:.4f}, POD: {pod_1mm:.4f}, FAR: {far_1mm:.4f}")
    print(f"4mm - CSI: {csi_4mm:.4f}, POD: {pod_4mm:.4f}, FAR: {far_4mm:.4f}")
    print(f"8mm - CSI: {csi_8mm:.4f}, POD: {pod_8mm:.4f}, FAR: {far_8mm:.4f}")
    print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    val()

