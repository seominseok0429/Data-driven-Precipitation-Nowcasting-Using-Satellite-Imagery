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

from torch.optim import Adam
import torch.nn.functional as F

from models.pix2pixhd  import Generator
from models.discriminator import Discriminator

from dataloader import Sat2RrdDataset 
from loss import GANLoss, SiLogLoss

from metrics import cal_csi
from utils import save_images


#python3 train.py --mid_type resnet --act_type relu --gan_loss True --cc_loss True --out_dir ./results_pix2pixcc/
#python3 train.py --mid_type resnet --act_type relu --gan_loss True --cc_loss False --out_dir ./results_pix2pixhd/
#python3 train.py --mid_type resnet --act_type relu --gan_loss False --cc_loss False --out_dir ./results_resUnet/

parser = argparse.ArgumentParser(description='PyTorch sat2rdr Training')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=8, type=float, help='cpu number')
parser.add_argument('--epoch', default=200, type=int, help='epoch')

parser.add_argument('--gpu_id', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--dataroot', default='./dataset/train/', type=str, help='data path')
parser.add_argument('--random_crop', default=True, type=bool, help='Enable random cropping of images.')
parser.add_argument('--random_hflip', default=True, type=bool, help='Enable random horizontal flipping of images.')
parser.add_argument('--random_vflip', default=True, type=bool, help='Enable random vertical flipping of images.')
parser.add_argument('--channels', default='0 1 2 3', type=str, help='List of channel indices to use.')

parser.add_argument('--in_ch', default=4, type=int, help='input channels')
parser.add_argument('--out_ch', default=1, type=int, help='output channels')

parser.add_argument('--mid_type', default='resnet', type=str, help='[resnet, van, convnext, vit]')
parser.add_argument('--act_type', default='relu', type=str, help='relu, silu, gelu')

parser.add_argument('--gan_loss', default=True, type=bool, help='Enable gan_loss')
parser.add_argument('--cc_loss', default=True, type=bool, help='Enable gan_loss')

parser.add_argument('--pixel_loss', default='no', type=str, help='no, mse, mae')

parser.add_argument('--out_dir', default='./results_resUnet/', type=str, help='result path')
parser.add_argument('--exp_name', default='', help='identifier for experiment')

args = parser.parse_args()

cudnn.benchmark = True

args.channels = [int(i) for i in args.channels.split(' ')]
train_dataset = Sat2RrdDataset(dataroot=args.dataroot, phase='train', random_crop=args.random_crop, random_hflip=args.random_hflip, random_vflip=args.random_vflip, channels=args.channels)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

val_dataset = Sat2RrdDataset(dataroot='./dataset/val/', phase='val', channels=args.channels)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

model_G = Generator(input_ch=args.in_ch, act_type=args.act_type, mid_type=args.mid_type)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.gan_loss:
    model_D = Discriminator(input_ch=args.in_ch, output_ch=args.out_ch)
    model_D.to(device)
    if device == 'cuda':
        model_D = torch.nn.DataParallel(model_D)

    model_D_optim = torch.optim.Adam(model_D.parameters(), lr=args.lr, betas=(0.5, 0.999))

model_G.to(device)

if device == 'cuda':
    model_G = torch.nn.DataParallel(model_G)
    cudnn.benchmark = True


if args.gan_loss:
    loss = GANLoss(args)
else:
    loss = nn.MSELoss()

model_G_optim = torch.optim.Adam(model_G.parameters(), lr=args.lr, betas=(0.5, 0.999))

train_total_iters = args.epoch * len(train_dataloader) # n_epochs * len(data_loader)
val_total_iters = len(val_dataloader)

out_dir = os.path.join(args.out_dir, args.exp_name)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def train(epoch):
    epoch_loss_G = 0
    epoch_loss_D = 0

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.epoch}", leave=False)

    for idx, batchs in pbar:
        if args.gan_loss:
            model_D.train()

        model_G.train()

        inputs, targets = batchs['A'].to('cuda:{}'.format(args.gpu_id)), batchs['B'].to('cuda:{}'.format(args.gpu_id))

        if args.gan_loss:
            loss_D, loss_G, target, fake = loss(model_D, model_G, inputs, targets) # D, G, input, target

            model_G_optim.zero_grad()
            loss_G.backward()
            model_G_optim.step()

            model_D_optim.zero_grad()
            loss_D.backward()
            model_D_optim.step()


            iters = epoch * len(train_dataloader) + idx

            if epoch > 100:
                lr = args.lr * (1 - iters / train_total_iters) ** 0.9

                model_G_optim.param_groups[0]["lr"] = lr
                model_D_optim.param_groups[0]["lr"] = lr
            else: 
                lr = args.lr

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()

            pbar.set_postfix({"Loss_G": f"{loss_G.item():.4f}", "Loss_D": f"{loss_D.item():.4f}", "LR": f"{lr:.14f}"})

        else:
            pred = model_G(inputs)

            loss_G = pixel_loss(pred, targets)

            model_G_optim.zero_grad()
            loss_G.backward()
            model_G_optim.step()

            iters = epoch * len(train_dataloader) + idx
           
            if epoch > 100:
                lr = args.lr * (1 - iters / train_total_iters) ** 0.9

                model_G_optim.param_groups[0]["lr"] = lr
            else: 
                lr = args.lr
            
            epoch_loss_G += loss_G.item()
        
            pbar.set_postfix({"Loss_G": f"{loss_G.item():.4f}", "LR": f"{lr:.14f}"})

    if (epoch+1)%10 ==0:
        torch.save(model_G.state_dict(), '%s/%d.pth' % (out_dir, (epoch+1)))

def val(epoch):

    csi_1mm, pod_1mm, far_1mm = 0, 0, 0
    csi_4mm, pod_4mm, far_4mm = 0, 0, 0
    csi_8mm, pod_8mm, far_8mm = 0, 0, 0

    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Epoch {epoch+1}/{args.epoch}", leave=False)

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
            
            csi, pod, far = cal_csi(pred, targets, threshold=1.0)
            csi_1mm += csi
            pod_1mm += pod
            far_1mm += far

            csi, pod, far = cal_csi(pred, targets, threshold=4.0)
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
    # epoch, input_image, true_image, pred_image, A_path, B_path
    #save_images(out_dir, epoch, inputs[-1, 0, :,:], targets[-1,0,:,:], pred[-1,0,:,:], A_path[-1], B_path[-1])
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

    print(f"Validation Results Epoch [{epoch+1}/200]:")
    print(f"1mm - CSI: {csi_1mm:.4f}, POD: {pod_1mm:.4f}, FAR: {far_1mm:.4f}")
    print(f"4mm - CSI: {csi_4mm:.4f}, POD: {pod_4mm:.4f}, FAR: {far_4mm:.4f}")
    print(f"8mm - CSI: {csi_8mm:.4f}, POD: {pod_8mm:.4f}, FAR: {far_8mm:.4f}")
    print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    for epoch in range(args.epoch):
        train(epoch)
        val(epoch)

