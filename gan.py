# -----------------------------------------------------------------------------
# import packages
# -----------------------------------------------------------------------------
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from torch.autograd import Variable

import sys, os, time, datetime, math
import numpy as np
import openpyxl
import argparse
import visdom
from PIL import Image

from networks import *
from noise_data import *
from DataSampler import *
# -----------------------------------------------------------------------------
# parameters from the argument 
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', default='resnet32', type=str, help='choose a type of model')
parser.add_argument('--dataset', default='mnist', type=str, help='choose a dataset')
parser.add_argument('--data_path', default='../../data/noise', type=str, help='path to data directory')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr_initial', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--lr_final', default=1e-4, type=float, help='final learning rate ')
parser.add_argument('--weight_decay',   type=float, default=0.0)
parser.add_argument('--momentum',       type=float, default=0.0)
parser.add_argument('--result_dir', default='../result/', type=str, help='directory of test dataset')
parser.add_argument('--iteration', default=1, type=int, help='number of trials')
parser.add_argument('--moreInfo', default=None, type=str, help='add more information on trial to the visdom window')

args    = parser.parse_args()
model_name      = args.model_name
dataset		    = args.dataset
data_path       = args.data_path
batch_size      = args.batch_size
epochs          = args.epochs
lr_initial      = args.lr_initial
lr_final        = args.lr_final
weight_decay    = args.weight_decay
momentum        = args.momentum
iteration       = args.iteration
moreInfo        = args.moreInfo

now             = datetime.datetime.now()
time_stamp      = now.strftime('%F_%H_%M_%S')

def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).detach()
    interpolates.requires_grad = True
    #interpolates = interpolates.cuda()

    #interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda())[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

def train():
    
    police.train()
    thief.train()
    
    G = []
    D = []
    
    # for idx, (data, target) in enumerate(loader_train):
    for idx, data in enumerate(loader_train):
        G_optimizer.zero_grad()

        # data, target = data.cuda(), target.cuda()
        # data = data.cuda()

        ####### type of data and noise ########
        # data = data.view(len(data),-1)
        # noise = torch.randn(len(data),100).cuda()
        
        data = data.cuda()
        noise = torch.randn(len(data),100).cuda()

        # train discriminator D

        # loss = -torch.sum(torch.log(police(data))) / len(data) - torch.sum(torch.log(1-police(thief(noise)))) / len(data)
        # loss.backward()

        for it in range(1):
            D_real = police(data).mean()
            with torch.no_grad():
                fake = thief(noise).detach()
            D_fake = police(fake).mean()
            gradient_penalty = calc_gradient_penalty(police, data, thief(noise))
            D_loss = D_fake - D_real + gradient_penalty
            G_optimizer.zero_grad()
            D_optimizer.zero_grad()
            D_loss.backward()
        
            D_optimizer.step()
            D.append(D_real.detach().cpu().numpy())

        # train generator G
        # loss = -torch.sum(torch.log(police(thief(noise)))) / len(data)

        for p in police.parameters():
            p.requires_grad = False  # to avoid computation
        noise = torch.randn(len(data),100).cuda()
        G_loss = -police(thief(noise)).mean()
        G_optimizer.zero_grad()
        D_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        G.append(G_loss.detach().cpu().numpy())
        for p in police.parameters():
            p.requires_grad = True  # to avoid computation

    
        # print(D_score_mean, G_score_mean, gradient_penalty.item())

    D_score_mean = np.mean(D)
    G_score_mean = np.mean(G)

    return {'D_loss_mean': D_score_mean, 'G_loss_mean': G_score_mean} 

#------------------------------------------------------------------------------
# load data
#------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor()])
                        
# set_train   = datasets.MNIST(root = data_path, train = True, download = True ,transform = transform)
# set_test    = datasets.MNIST(root = data_path, train = False, download = True ,transform = transform)
set_train = noise_set(root=data_path, noise_type=dataset, transform=transform)
#print(set(np.sort(set_train[0].flatten()).tolist()))
#assert(0 == 1)

loader_train = torch.utils.data.DataLoader(set_train, batch_size = batch_size, shuffle = True, drop_last=True)

# sample_indices = getSampleIndices(set_train, 0.5)
# train_sampler   = torch.utils.data.SubsetRandomSampler(sample_indices)    # when using only fraction of training dataset
# loader_train    = torch.utils.data.DataLoader(set_train, batch_size = batch_size, sampler = train_sampler, drop_last=True)

# loader_test     = torch.utils.data.DataLoader(set_test, batch_size = batch_size, shuffle = False, drop_last=True)

police = discriminator().cuda()
thief  = generator().cuda()
    
# G_optimizer   = optim.SGD(thief.parameters(), lr = lr_initial, momentum = momentum, weight_decay = weight_decay)
# D_optimizer   = optim.SGD(police.parameters(), lr = lr_initial, momentum = momentum, weight_decay = weight_decay) 
G_optimizer   = optim.Adam(thief.parameters(), lr=lr_initial, betas=(0.5, 0.9))
D_optimizer   = optim.Adam(police.parameters(), lr=lr_initial, betas=(0.5, 0.9))

if(lr_initial != lr_final):
    G_scheduler = optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[80,120,160], gamma=0.5)
    D_scheduler = optim.lr_scheduler.MultiStepLR(D_optimizer, milestones=[80,120,160], gamma=0.5)

D_loss = np.zeros(epochs)
G_loss = np.zeros(epochs)

# -----------------------------------------------------------------------------
# iteration for the epoch
# -----------------------------------------------------------------------------

for e in range(epochs):

    time_start  = time.time()

    if(lr_initial != lr_final):

        G_scheduler.step()
        D_scheduler.step()

    #   if e >= smoothing_start - 1 + int(interval/2):
    #       for param_group in optimizer.param_groups:
    #            param_group['lr'] = 0.001

    result_train    = train()

    with torch.no_grad():
        if ((e+1)==5): # make result directory
            result_directory = args.result_dir + time_stamp + '_' + moreInfo
            os.makedirs(result_directory)

        if ((e+1)%5 == 0):
            thief.eval()
            # noise = torch.randn(10,100).cuda()
            # result_img = thief(noise).view(len(noise), 1, 28, 28)

            # noise = torch.randn(4,1,16,16).cuda()
            # result_img = thief(noise)
            noise = torch.randn(8,100).cuda()
            result_img = thief(noise).view(len(noise), 1, 64, 64)
            save_image(result_img, result_directory + '/epoch_'+str(e+1)+'.jpg', nrow=4)

    D_loss[e]   = result_train['D_loss_mean']
    G_loss[e]   = result_train['G_loss_mean']

    time_elapsed    = time.time() - time_start # in seconds

    for param_group in D_optimizer.param_groups:

        learning_rate = param_group['lr']

    print('epoch: {:3d}/{:3d}, lr: {:6.5f}, D_loss(mean): {:8.5f}, G_loss(mean): {:8.5f} ({:.0f}s) '
            .format(e+1, epochs, learning_rate, D_loss[e], G_loss[e], time_elapsed))
            
    #
    # early termination
    #
    #if (e + 1) > 20 and accuracy_test[e,iter] < 50: break
