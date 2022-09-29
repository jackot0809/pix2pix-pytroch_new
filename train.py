#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn
import os
import time
from torchvision.utils import save_image

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset
from models import VGGLoss
from models import MultiscaleDiscriminator
from models import PredictionNetwork
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--display', type=int, default=5, help='display frequency')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
parser.add_argument('--netG', type=str, default='checkpoint/7161/netG_A2B_5.pth', help='generator checkpoint file')
parser.add_argument('--netD', type=str, default='checkpoint/7161/netD_A_5.pth', help='netD checkpoint file')

opt = parser.parse_args()
print(opt)
print('start from {} epoch'.format(opt.epoch))

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
    
# Networks
netG= Generator(opt.input_nc, opt.output_nc)
netD = MultiscaleDiscriminator(opt.input_nc + opt.output_nc, opt.ndf, opt.n_layers_D, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, num_D=1, getIntermFeat=False)   
netP = PredictionNetwork(input_nc=opt.input_nc,output_nc=opt.output_nc, num_downs=7 ,ngf=32,norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=3)

if opt.cuda:
    netG.cuda()
    netD.cuda()
    netP.cuda()

netG.apply(weights_init_normal)
netD.apply(weights_init_normal)
netP.apply(weights_init_normal)

####Load state dicts
# netG.load_state_dict(torch.load(opt.netG))
# netD.load_state_dict(torch.load(opt.netD))
# print("load {} epoch model".format(opt.generator_A2B.split('/')[-1].split('.')[0].split('_')[-1]))

# Loss
criterion_GAN = torch.nn.MSELoss()
criterion_l1 = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(netG.parameters(),lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_P = torch.optim.Adam(netP.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D= torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_P= torch.optim.lr_scheduler.LambdaLR(optimizer_P, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
# input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
# input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
# input_C = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

# Dataset loader
transforms_ = [
                transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                transforms.ToTensor()]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.epoch, opt.n_epochs, len(dataloader),opt.display)
###################################


# Create checkpoint dirs if they don't exist
checkpoint_path='checkpoint/'+opt.dataroot.split('/')[-1]
# print(checkpoint_path)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Create medium dirs if they don't exist
medium_path='medium/'+opt.dataroot.split('/')[-1]
# print(medium_path)
if not os.path.exists(medium_path):
    os.makedirs(medium_path)

###
def GANloss(predict, target_is_real):
    loss=0
    # print(len(predict[0]))
    for pred in predict[0]:
        if target_is_real:
            target = Variable(Tensor(pred.size()).fill_(1.0), requires_grad=False)
        else:
            target = Variable(Tensor(pred.size()).fill_(0.0), requires_grad=False)
        loss += criterion_GAN(pred, target) 
    return loss

###
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3] 
    
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    start=time.time()
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = batch['A'].cuda()
        real_B = batch['B'].cuda()
        real_A1 = batch['A1'].cuda()
        real_B1 = batch['B1'].cuda()

        fake_B = netG(real_A)
        fake_B1 = netG(real_A1)
        ###### Discriminator ######
        optimizer_D.zero_grad()

        # Real loss
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = netD(real_AB.detach())
        loss_D_realA0 = GANloss(pred_real, True)

        real_AB2 = torch.cat((real_A1, real_B1), 1)
        pred_real2 = netD(real_AB2.detach())
        loss_D_realA1 = GANloss(pred_real2, True)

        loss_D_real = loss_D_realA0 + loss_D_realA1

        # Fake loss
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = netD(fake_AB.detach())
        loss_D_fakeA0 = GANloss(pred_fake, False)
        
        fake_AB2 = torch.cat((real_A1, fake_B1), 1)
        pred_fake2 = netD(fake_AB2.detach())
        loss_D_fakeA1 = GANloss(pred_fake2, False)

        # fake_prediction_AB = torch.cat((real_B2, fake_prediction_B2), 1)
        # pred_fake3 = netD(fake_prediction_AB.detach())
        # loss_D_prediction = GANloss(pred_fake3, False)
        
        loss_D_fake = loss_D_fakeA0 + loss_D_fakeA1  #+ loss_D_prediction

        # Total loss
        loss_D = (loss_D_real + loss_D_fake)*0.5

        loss_D.backward()

        optimizer_D.step()
        ###################################  

        ###### Generator ######
        optimizer_G.zero_grad()
        optimizer_P.zero_grad()

        # GAN loss
        fake_AB1 = torch.cat((real_A, fake_B), 1)
        pred_fake = netD(fake_AB1.detach())
        loss_GAN = GANloss(pred_fake, True)
        loss_l1 = criterion_l1(fake_B, real_B)*10.0

        fake_AB2 = torch.cat((real_A1, fake_B1), 1)
        pred_fake2 = netD(fake_AB2.detach())
        loss_GAN += GANloss(pred_fake2, True)
        loss_l1 += criterion_l1(fake_B1, real_B1)*10.0
        
        real_B2 = batch['B2'].cuda()
        fake_prediction_B2 = netP(fake_B, fake_B1) #fake_prediction_frame
        loss_l1 += criterion_l1(fake_prediction_B2, real_B2)*10.0

        # Total loss
        loss_G = loss_GAN + loss_l1
        loss_G.backward()
        
        optimizer_G.step()
        optimizer_P.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_GAN': loss_GAN,
                    'loss_l1': loss_l1, 
                    'loss_D': loss_D},
                    images={
                            'real_A_frame1': real_A, 'fake_B_frame1': fake_B, 'real_B_frame1': real_B,
                            'real_A_frame2': real_A1, 'fake_B_frame2': fake_B1, 'real_B_frame2': real_B1,
                            'fake_next_frame3': fake_prediction_B2, 'real_next_frame3':real_B2})

                            
    last=time.time()
    print("time per epoch: {}s".format(last-start))
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    lr_scheduler_P.step()

    # Save models checkpoints
    if ((epoch+1)%5 ==0):
        torch.save(netG.state_dict(), checkpoint_path+'/netG_{}.pth'.format(epoch+1))
        torch.save(netD.state_dict(), checkpoint_path+'/netD_{}.pth'.format(epoch+1))
        torch.save(netP.state_dict(), checkpoint_path+'/netD_{}.pth'.format(epoch+1))

    # Save intermedium output
    if ((epoch+1)%2 ==0):
        real_A=0.5*(real_A+1.0)
        fake_B=0.5*(fake_B+1.0)  
        save_image(real_A[0], medium_path+'/{}_real_A.jpg'.format(epoch+1),padding=0)    
        save_image(fake_B[0], medium_path+'/{}_fake_B.jpg'.format(epoch+1),padding=0)    
    
###################################
