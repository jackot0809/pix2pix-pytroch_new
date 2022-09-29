#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import cv2
import numpy as np

from models import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_G', type=str, default='checkpoint/json_train_testing/netG_25.pth', help='generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######

# Networks
netG = Generator(opt.input_nc, opt.output_nc)

if opt.cuda:
    netG.cuda()

# Load state dicts
netG.load_state_dict(torch.load(opt.generator_G))

# Set model's test mode
netG.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader integer division or modulo by zero

transforms_ = [ transforms.ToTensor(),
                ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
fakeB_path='output/'+opt.dataroot.split('/')[-1]+'/fake_B'

if not os.path.exists(fakeB_path):
    os.makedirs(fakeB_path)

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = batch['A'].cuda()
    filename= batch['filename'][0]
    
    
    # Generate output
    fake_B = 0.5*(netG(real_A) + 1.0)
    fake_B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(fake_B)

    k=0
    save_image(fake_B, fakeB_path+'/{}.jpg'.format(filename),padding=0)    

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
