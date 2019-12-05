# -*- coding: utf-8 -*-
"""shock_net

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QOKgPExfXB3FGTKP2xLru1k3cCjYWPcM

# ToDo List
* よい初期値を探す（conv, convtransposed)
* rho,u,v

---
* distance functioniを作成する
* distance indexをlossの中に加える
* 損失関数を作成する(衝撃波部分＋オイラー方程式部分）
"""

from google.colab import drive
drive.mount('/content/drive')

import shutil

shutil.copytree("/content/drive/My Drive/shape_to_shock/", "/content/ss")

import os

os.chdir("ss")

ls
# -*- coding: utf-8 -*-
"""
Shcok Net
==============

**Author**: Taro Kawasaki

"""

from __future__ import print_function

import glob
import os.path as osp

import numpy as np
import torch.utils.data as data

# %matplotlib inline
import random
import math
from tqdm import tqdm

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torch.nn.functional as Fun


from IPython.display import HTML

"""# HyperParameters"""

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


######################################################################
# fluid parameters
# ------

# specific heat ratio
gamma=1.4

######################################################################
# NN parameters
# ------

# Batch size during training
batch_size = 2

# input size
input_size=200

# Number of training epochs
num_epochs = 500

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = torch.cuda.device_count()
print("Let's use", ngpu, "GPUs!")

# Number of conv output channels
conv_channel1 = 4

# Number of conv output channels
conv_channel2 = 8

# Number of conv output channels
conv_channel3 = 16

# Number of hidden layer
n_hidden= 10

# Number of channels for output
output_channel= 4

# weight for prediction loss
lambda_predction= 0.1

# weight for conservation loss
lambda_conservation= 0.1

# weight for RH loss
lambda_rh= 0.1

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""# Data Loader"""

def make_data_path_list(phase="train"):
    """

    Parameters
    ----------
    phase : 'train' or 'val'

    Returns
    -------
    path_list : list
    """

    #rootpath = "./data/"
    rootpath = "./data/"
    target_path = osp.join(rootpath + phase + '/**/*/')
    #print(target_path)
    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)
    #print(path_list)
    return path_list


class Dataset(data.Dataset):
    """

    Attributes
    ----------
    file_list : list
    transform : object
    phase : 'train' or 'test'
    """

    def __init__(self, file_list, phase='train'):
        self.file_list = file_list  # file path
        # self.transform = transform  #
        self.phase = phase  # train or val
        self.size = input_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        read_list = ["rho", "u","v","pressure"]

        ##########
        # Input
        input_labels = []

        ##########
        # Output
        fluid= []
        shock_labels= []

        data_path = self.file_list[index]

        delta_x=random.randrange(256-self.size)
        delta_y=random.randrange(256-self.size)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
        # distance
        ### 転置が掛かってるかもだけど、なんかおかしい。向きに注意

        data_path_distance="/".join(data_path.split("/")[:4])
        data_path_distance=data_path_distance+"/distance.csv"
        data_distance= np.loadtxt(data_path_distance, delimiter=",")
        #data_distance=data_distance.T


        data_distance=data_distance[delta_y:self.size+delta_y,delta_x:self.size+delta_x]
        data_distance = np.reshape(data_distance, (1,self.size, self.size))

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
        # boundary

        boundary=abs(data_distance)>=1e-4
        boundary=boundary.astype(int)
        boundary= np.reshape(boundary, (1,self.size, self.size))

        # plt.imshow(shape_index)
        # plt.show()

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
        # input_labels
        temp_label = data_path.split("/")[4].split("_")
        input_labels.append(float(temp_label[0][4:]))
        input_labels.append(float(temp_label[1][5:]))

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
        # fluid 
        for item in read_list:
            path = osp.join(data_path + "/" + item + ".csv")
            data = np.loadtxt(path, delimiter=",")
            data = data.T
            data=data[delta_y:self.size+delta_y,delta_x:self.size+delta_x]
            data = np.reshape(data, (self.size, self.size))
            #if item=="pressure":
                #shape_index=abs(data)<=1e-4
                #fluid.append((shape_index.astype(int)))
                #plt.imshow(shape_index.astype(int),interpolation="nearest")
                #plt.show()
            fluid.append(data)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
        # shock_labels
        shock_path = osp.join(data_path + "/" + "result"+ ".csv")

        shock= np.loadtxt(shock_path, delimiter=",")
        shock=shock.T

        shock=shock[delta_y:self.size+delta_y,delta_x:self.size+delta_x]
        shock= np.reshape(shock, (self.size, self.size))
        shock_labels.append(shock)


        boundary=np.array(boundary,dtype="float64")
        distance=np.array(data_distance,dtype="float64")
        input_labels=np.array(input_labels,dtype="float64")
        fluid=np.array(fluid,dtype="float64")
        shock_labels=np.array(shock_labels,dtype="float64")


        return boundary,distance,input_labels,fluid,shock_labels


def load(phase, batch_size):
    if phase == "train":
        train_dataset = Dataset(file_list=make_data_path_list("train"), phase="train")
        train_dataloader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        return train_dataloader
        
    if phase == "val":
        train_dataset = Dataset(file_list=make_data_path_list("train"), phase="train")
        train_dataloader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        return train_dataloader

######################################################################
# Data
# Create the dataloader

train_dataloader=load("train",batch_size=batch_size)

for i in train_dataloader:
    a,b,c,d,e=i
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    print(e.shape)

#while(True):
#    for i in train_dataloader:
#        print("")

######################################################################
# Implementation
# --------------
#
def weights_init(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m)==nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m)==nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif type(m)==nn.ConvTranspose2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


    #classname = m.__class__.__name__
    #if classname.find('Conv') != -1:
    #    nn.init.normal_(m.weight.data, 0.0, 0.02)
    #elif classname.find('BatchNorm') != -1:
    #    nn.init.normal_(m.weight.data, 1.0, 0.02)
    #    nn.init.constant_(m.bias.data, 0)

######################################################################
# Encoder
# ~~~~~~~~~
# Encoder Code
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.channel =1
        self.model=nn.Sequential(
            # layer 1
            nn.Conv2d(self.channel, conv_channel1, kernel_size=4, stride=4, bias=True),
            nn.BatchNorm2d(conv_channel1),
            nn.LeakyReLU(0.1),

            # layer 2
            nn.Conv2d(conv_channel1, conv_channel2, kernel_size=4, stride=4, bias=True),
            nn.BatchNorm2d(conv_channel2),
            nn.LeakyReLU(0.1),

            # layer 3
            nn.Conv2d(conv_channel2, conv_channel3, kernel_size=4, stride=4, bias=True),
            nn.BatchNorm2d(conv_channel3),
            nn.LeakyReLU(0.1),

        )
    def forward(self,input):
        return self.model(input)

######################################################################
# concate
# ~~~~~~~~~
# Linear Code

class Linear(nn.Module):
    def __init__(self):
        super(Linear,self).__init__()
        self.n_hidden=n_hidden
        self.size=3
        self.model=nn.Sequential(
            # layer 1
            nn.Linear(conv_channel3*self.size**2+2,self.n_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(self.n_hidden,conv_channel3*self.size**2)
        )
            
    def forward(self,input):
        return torch.reshape(self.model(input),(-1,conv_channel3,self.size,self.size))

######################################################################
# Decoder
# ~~~~~~~~~
# Decoder Code
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.channel = output_channel
        self.model=nn.Sequential(
            # layer 1
            nn.ConvTranspose2d(conv_channel3, conv_channel2, kernel_size=4, stride=4, bias=True),
            nn.BatchNorm2d(conv_channel2),
            nn.LeakyReLU(0.1),

            # layer 2
            nn.ConvTranspose2d(conv_channel2, conv_channel1, kernel_size=6, stride=4, bias=True),
            nn.BatchNorm2d(conv_channel1),
            nn.LeakyReLU(0.1),
#
            # layer 3
            nn.ConvTranspose2d(conv_channel1,self.channel, kernel_size=4, stride=4, bias=True),
            nn.BatchNorm2d(self.channel),
            nn.ReLU(True)

        )
    def forward(self,input):
        return self.model(input)

"""# Unet"""

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = Fun.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.outconv=nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.Sigmoid()
        )
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        #return self.conv(x)
        return self.outconv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        shock_prediction= self.outc(x)
        return shock_prediction

"""# ShockNet"""

######################################################################
# ShockNet
# ~~~~~~~~~
# Shockt Code
class ShockNet(nn.Module):
    def __init__(self):
        super(ShockNet,self).__init__()
        self.encoder=Encoder()
        self.linear=Linear()
        self.decoder=Decoder()
        self.unet=UNet()

    def _forward(self,distance,input_labels):

        # Encoder
        x=self.encoder(distance)
        
        feature_1=x.shape[1]
        feature_2=x.shape[2]
        feature_3=x.shape[3]
        

        # Hidden
        x=torch.reshape(x,(-1,feature_1*feature_2*feature_3))
        x=torch.cat((x,input_labels),1)
        x=self.linear(x)

        # Decoder
        x=self.decoder(x)
        values=x


        # Shock Detection
        logits=self.unet(x)
        
        return values,logits

    def forward(self,distance,input_labels):
        x=self._forward(distance,input_labels)

        return x

"""# Tensorboard"""

#from torch.utils.tensorboard import SummaryWriter
#
## default `log_dir` is "runs" - we'll be more specific here
#writer = SummaryWriter('runs/')

#!pip install tensorboard==1.14.0

#!pip install tb-nightly

#writer.add_graph(shock_net,np.zeros((256,256)))
#writer.close()

#%load_ext tensorboard
#%tensorboard --logdir ./runs]



"""# Distance Function"""

# def sample_func(process_index):
#     #print('process index: %s started.' % process_index)
#     num = 0
# 
#     for i in range(1000000000):
#         num += 1
#     #print('process index: %s ended.' % process_index)

#from multiprocessing import Process
#process_list = []
#for i in tqdm(range(10)):
#    process = Process(
#        target=sample_func,
#        kwargs={'process_index': i})
#    process.start()
#    process_list.append(process)
#
#for process in process_list:
#    process.join()

def distance_function(input):
    shape_index=abs(input)<=1e-4
    distance=shape_index.astype(int)
    index=np.where(distance==0)
    index=np.stack([index[0],index[1]],1)

    for i in tqdm(range(0,distance.shape[0])):
        for j in range(0,distance.shape[1]):
            temp=[]
            for item in index:
                x=i-item[0]
                y=j-item[1]
                temp.append(math.sqrt(x**2+y**2))
    plt.imshow(distance)
    plt.show()
    return distance

"""# Loss Function"""

list=[]
list.append([1,2,3])
list.append([4,5,6])
a=np.array(list)

a=a**2
print(a)
print(np.log(a))

np.ones(4).shape

"""## Entropy Difference"""

def sinBeta(rho,u,v,p):
    # p 
    p_1=p[:,:input_size-1,:]+1e-5
    p_2=p[:,1:input_size,:]+1e-5

    # u
    u_1=u[:,:input_size-1,:]+1e-5

    # v
    v_1=v[:,:input_size-1,:]+1e-5

    # rho
    rho_1=rho[:,:input_size-1,:]+1e-5

    # M1^2
    M_1_sq=rho_1*(u_1**2+v_1**2)/(gamma*p_1)+1e-5


    # sin Bet
    sinBeta=(p_2*(gamma+1)/p_1+gamma-1)/(2*gamma*M_1_sq)+1e-5

    return sinBeta,M_1_sq

#def deltaEntropy(rho,u,v,p):
#    # p 
#    p_1=p[:,:input_size-1,:]+1e-5
#    p_2=p[:,1:input_size,:]+1e-5
#
#    # u
#    u_1=u[:,:input_size-1,:]+1e-5
#
#    # v
#    v_1=v[:,:input_size-1,:]+1e-5
#
#    # rho
#    rho_1=rho[:,:input_size-1,:]+1e-5
#
#    # M1^2
#    M_1_sq=rho_1*(u_1**2+v_1**2)/(gamma*p_1)+1e-5
#
#
#    # sin Bet
#    sinBeta=(p_2*(gamma+1)/p_1+gamma-1)/(2*gamma*M_1_sq)+1e-5
#
#    temp=gamma/(gamma-1)*torch.log(1e-5+ ((gamma-1)*M_1_sq*sinBeta+2) /(1e-5+(gamma+1)*M_1_sq*sinBeta))
#    temp2=(1/(gamma-1))*torch.log(1e-5+ (2*gamma*M_1_sq*sinBeta-(gamma-1))/(gamma+1) )
#
#    return temp+temp2



def RHLoss(fluid_prediction,shock_prediction,fluid_gt,shock_gt,boundary):

    boundary_gt=boundary[:,:,input_size-1,:]
    # prediction
    rho_prediction=fluid_prediction[:,0,:,:]
    u_prediction=fluid_prediction[:,1,:,:]
    v_prediction=fluid_prediction[:,2,:,:]
    p_prediction=fluid_prediction[:,3,:,:]

    _,M_1_sq_pre=sinBeta(rho_prediction,u_prediction,v_prediction,p_prediction)

    # ground truth
    rho_gt=fluid_gt[:,0,:,:]
    u_gt=fluid_gt[:,1,:,:]
    v_gt=fluid_gt[:,2,:,:]
    p_gt=fluid_gt[:,3,:,:]

    p_1=p_gt[:,:input_size-1,:]+1e-5
    p_2=p_gt[:,1:input_size,:]+1e-5
    left_hand_side=p_2/p_1

    left_hand_side=left_hand_side*shock_gt[:,:,input_size-1,:]
    left_hand_side=torch.where(abs(left_hand_side)<6,left_hand_side,p_1)


    sinBeta_gt,_=sinBeta(rho_prediction,u_prediction,v_prediction,p_prediction)
    right_hand_side=(2*gamma*M_1_sq_pre*sinBeta_gt-(gamma-1))/(gamma+1)*shock_prediction[:,:,input_size-1,:]
    right_hand_side=torch.where(abs(right_hand_side)<6,right_hand_side,p_1)
    #print(torch.max(right_hand_side))


    loss= (left_hand_side-right_hand_side)**2
    #loss=loss*boundary_gt

    return loss.sum()/(p_1.shape[0]*p_1.shape[1]*p_1.shape[2])
    #return torch.sum(loss)


#
#def EntropyLoss(fluid_prediction,shock_prediction,fluid_gt,shock_gt,boundary):
#
#    boundary_gt=boundary[:,:,input_size-1,:]
#    # prediction
#    rho_prediction=fluid_prediction[:,0,:,:]
#    u_prediction=fluid_prediction[:,1,:,:]
#    v_prediction=fluid_prediction[:,2,:,:]
#    p_prediction=fluid_prediction[:,3,:,:]
#
#
#    #print(torch.argmax(deltaEntropy(rho_prediction,u_prediction,v_prediction,p_prediction)))
#
#    deltaE_prediction=deltaEntropy(rho_prediction,u_prediction,v_prediction,p_prediction)*shock_prediction[:,:,input_size-1,:]
#
#    # ground truth
#    rho_gt=fluid_gt[:,0,:,:]
#    u_gt=fluid_gt[:,1,:,:]
#    v_gt=fluid_gt[:,2,:,:]
#    p_gt=fluid_gt[:,3,:,:]
#
#    deltaE_gt=deltaEntropy(rho_gt,u_gt,v_gt,p_gt)*shock_gt[:,:,input_size-1,:]
#    loss= (deltaE_gt-deltaE_prediction)**2
#    loss=loss*boundary_gt
#
#    return loss.sum()/(rho_gt.shape[0]*rho_gt.shape[1]*rho_gt.shape[2])
#    #return torch.sum(loss)
#
#
#

#a=torch.from_numpy(np.random.rand(2,4,200,200))
#b=torch.from_numpy(np.random.rand(2,1,200,200))
#c=torch.from_numpy(np.random.rand(2,4,200,200))
#d=torch.from_numpy(np.random.rand(2,1,200,200))
#
#print(EntropyLoss(a,b,c,d).item())

"""## Flux Vector Spliting"""

def Q_value(rho,u,v,p,phase="default"):
    rho=rho.to(device)
    u=u.to(device)
    v=v.to(device)
    p=p.to(device)
    e=p/(gamma-1)+0.5*rho*(u*u+v*v)
    e=torch.where(e<10,e,p)
    if phase=="default":
        return [rho, rho*u, rho*v,e]
    else:
        return rho+rho*u+ rho*v+e



def E_value(rho,u,v,p,phase="default"):
    rho=rho.to(device)
    u=u.to(device)
    v=v.to(device)
    p=p.to(device)
    e=p/(gamma-1)+0.5*rho*(u*u+v*v)
    e=torch.where(e<10,e,p)
    if phase=="default":
        return[ rho*u, p+rho*u*u, rho*u*v, (e+p)*u]

    else:
        return rho*u+p+rho*u*u+rho*u*v+(e+p)*u


def F_value(rho,u,v,p,phase="default"):
    rho=rho.to(device)
    u=u.to(device)
    v=v.to(device)
    p=p.to(device)
    e=p/(gamma-1)+0.5*rho*(u*u+v*v)
    e=torch.where(e<10,e,p)
    if phase=="default":
        return[ rho*v, rho*u*v, p+rho*v*v, (e+p)*v]

    else:
        return rho*v+rho*u*v+p+rho*v*v+(e+p)*v


def AB_value(rho,u,v,p):
    rho=rho+1e-10
    u=u+1e-10
    v=v+1e-10
    p=p+1e-10

    e=p/(gamma-1)+0.5*rho*(u*u+v*v)
    q=(u**2+v**2)
    listA=[]
    #listA.append([np.zeros(rho.shape),np.ones(rho.shape),np.zeros(rho.shape),np.zeros(rho.shape)])
    #listA.append([0.5*(gamma-3)*u*u + 0.5*(gamma-1)*v*v,-(gamma-3)*u,-(gamma-1)*v,(gamma-1)*np.ones(rho.shape)])
    #listA.append([-u*v,v,u,np.zeros(rho.shape)])
    #listA.append([-gamma*u*e/rho+(gamma-1)*u*q,gamma*e/rho-0.5*(gamma-1)*(2*u*2+q),-(gamma-1)*u*v,gamma*u])
    listA.append([torch.zeros(rho.shape),torch.ones(rho.shape),torch.zeros(rho.shape),torch.zeros(rho.shape)])
    listA.append([0.5*(gamma-3)*u*u + 0.5*(gamma-1)*v*v,-(gamma-3)*u,-(gamma-1)*v,(gamma-1)*torch.ones(rho.shape)])
    listA.append([-u*v,v,u,torch.zeros(rho.shape)])
    listA.append([-gamma*u*e/rho+(gamma-1)*u*q,gamma*e/rho-0.5*(gamma-1)*(2*u*2+q),-(gamma-1)*u*v,gamma*u])

    listB=[]
    #listB.append([np.zeros(rho.shape),np.zeros(rho.shape),np.ones(rho.shape),np.zeros(rho.shape)])
    #listB.append([-u*v,v,u,np.zeros(rho.shape)])
    #listB.append([0.5*(gamma-3)*v**2+0.5*(gamma-1)*u**2,-(gamma-1)*u,-(gamma-3)*v,(gamma-1*np.ones(rho.shape))])
    #listB.append([-gamma*v*e/rho+(gamma-1)*v*q,-(gamma-1)*u*v,gamma*e/rho+0.5*(gamma-1)*(2*v**2+q),gamma*v])

    listB.append([torch.zeros(rho.shape),torch.zeros(rho.shape),torch.ones(rho.shape),torch.zeros(rho.shape)])
    listB.append([-u*v,v,u,torch.zeros(rho.shape)])
    listB.append([0.5*(gamma-3)*v**2+0.5*(gamma-1)*u**2,-(gamma-1)*u,-(gamma-3)*v,(gamma-1*torch.ones(rho.shape))])
    listB.append([-gamma*v*e/rho+(gamma-1)*v*q,-(gamma-1)*u*v,gamma*e/rho+0.5*(gamma-1)*(2*v**2+q),gamma*v])
    list_AB=listA+listB

    return list_AB

import torch 
torch.cuda.memory_allocated()
torch.cuda.empty_cache()

torch.cuda.memory_cached()

def ConservationLoss(input,target):
    # MS Error
    loss=nn.MSELoss()(input,target)
    # Euler equations
    # FVS
    ### read_list = ["pressure", "mach","rho","temperature"]
    ### read_list = ["rho", "u","v","p"]
    

    rho=input[:,0,:,:]
    u=input[:,1,:,:]
    v=input[:,2,:,:]
    p=input[:,3,:,:]

    #print("maxx")
    #print(torch.max(rho))
    #print(torch.max(u))
    #print(torch.max(v))
    #print(torch.max(p))
    #print("rho sum")
    #print(rho.sum()) 

    # rho
    rho_x_jm1=rho[:,:input_size-2,:]
    rho_x_j=rho[:,1:input_size-1,:]
    rho_x_jp1=rho[:,2:input_size,:]
    
    rho_y_jm1=rho[:,:,:input_size-2]
    rho_y_j=rho[:,:,1:input_size-1]
    rho_y_jp1=rho[:,:,2:input_size]
    
    # u
    u_x_jm1=u[:,:input_size-2,:]
    u_x_j=u[:,1:input_size-1,:]
    u_x_jp1=u[:,2:input_size,:]
    
    u_y_jm1=u[:,:,:input_size-2]
    u_y_j=u[:,:,1:input_size-1]
    u_y_jp1=u[:,:,2:input_size]
    
    # v
    v_x_jm1=v[:,:input_size-2,:]
    v_x_j=v[:,1:input_size-1,:]
    v_x_jp1=v[:,2:input_size,:]
    
    v_y_jm1=v[:,:,:input_size-2]
    v_y_j=v[:,:,1:input_size-1]
    v_y_jp1=v[:,:,2:input_size]
    
    # p
    p_x_jm1=p[:,:input_size-2,:]
    p_x_j=p[:,1:input_size-1,:]
    p_x_jp1=p[:,2:input_size,:]
    
    p_y_jm1=p[:,:,:input_size-2]
    p_y_j=p[:,:,1:input_size-1]
    p_y_jp1=p[:,:,2:input_size]
    
    ##########
    
    # Q
    Q_jm1=Q_value(rho_x_jm1,u_x_jm1,v_x_jm1,p_x_jm1)

    Q_j=Q_value(rho_x_j,u_x_j,v_x_j,p_x_j)

    Q_jp1=Q_value(rho_x_jp1,u_x_jp1,v_x_jp1,p_x_jp1)

    for i in range(0,4):
        Q_j[i]= Q_j[i].type(torch.cuda.FloatTensor)
        Q_jp1[i]= Q_jp1[i].type(torch.cuda.FloatTensor)

    # E
    E_jm1_sum=E_value(rho_x_jm1,u_x_jm1,v_x_jm1,p_x_jm1,"sum")
    E_jp1_sum=E_value(rho_x_jp1,u_x_jp1,v_x_jp1,p_x_jp1,"sum")
    
    # F
    F_jm1_sum=F_value(rho_y_jm1,u_y_jm1,v_y_jm1,p_y_jm1,"sum")
    F_jp1_sum=F_value(rho_y_jp1,u_y_jp1,v_y_jp1,p_y_jp1,"sum")

    
    # AB
    AB_jm1=AB_value(rho_x_jm1,u_x_jm1,v_x_jm1,p_x_jm1)
    AB_j=AB_value(rho_x_j,u_x_j,v_x_j,p_x_j)
    AB_jp1=AB_value(rho_x_jp1,u_x_jp1,v_x_jp1,p_x_jp1)
    
    
    ##########
    for i in range(0,4):
        for j in range(0,4):
            #Q_jm1[i]= Q_jm1[i].type(torch.cuda.FloatTensor)
            #AB_jm1[j][i]= AB_jm1[j][i].type(torch.cuda.FloatTensor)
#
            #Q_j[i]= Q_j[i].type(torch.cuda.FloatTensor)
            #AB_j[j][i]= AB_j[j][i].type(torch.cuda.FloatTensor)
#
            #Q_jp1[i]= Q_jp1[i].type(torch.cuda.FloatTensor)
            #AB_jp1[j][i]= AB_jp1[j][i].type(torch.cuda.FloatTensor)

            try:
                ABQ_jp1=torch.from_numpy(abs(AB_jp1[j][i])).to(device) *Q_jp1[i]
                ABQ_jp1=torch.where(ABQ_jp1[j][i]<10.0,ABQ_jp1[j][i],Q_jp1[i])
            except:
                AB_jp1[j][i]= AB_jp1[j][i].to(device)
                Q_jp1[i]= Q_jp1[i].to(device)

                ABQ_jp1=AB_jp1[j][i] *Q_jp1[i]
                ABQ_jp1=torch.where(abs(ABQ_jp1).to(device)<100.0,ABQ_jp1.to(device),Q_jp1[i].to(device))
            

            try:
                ABQ_j=2*torch.from_numpy(abs(AB_j[j][i])).to(device)*Q_j[i]
                ABQ_j=torch.where(AB_j[i][j]<10.0,AB_j[i][j],Q_j[i])
            except:
                AB_j[j][i]= AB_j[j][i].to(device)
                Q_j[i]= Q_j[i].to(device)

                ABQ_j=2*abs(AB_j[j][i])*Q_j[i]
                ABQ_j=torch.where(abs(ABQ_j).to(device)<100.0,ABQ_j.to(device),Q_j[i].to(device)).to(device)

            try:
                ABQ_jm1=torch.from_numpy(abs(AB_jm1[j][i])).to(device)*Q_jm1[i]
                ABQ_jm1=torch.where(AB_jm1[i][j]<10.0,AB_jm1[i][j],Q_jm1[i])
            except:
                AB_jm1[j][i]= AB_jm1[j][i].to(device)
                Q_jm1[i]= Q_jm1[i].to(device)

                ABQ_jm1=abs(AB_jm1[j][i])*Q_jm1[i]
                ABQ_jm1=torch.where(abs(ABQ_jm1).to(device)<100.0,ABQ_jm1.to(device),Q_jm1[i].to(device)).to(device)
            

            #print("ABQ")
            #print(torch.max(ABQ_jp1))
            #print(torch.max(ABQ_j))
            #print(torch.max(ABQ_jm1))
            loss=loss-ABQ_jp1.sum()+ABQ_j.sum()-ABQ_jm1.sum()
    
    

    E_jp1_sum=torch.where(E_jp1_sum.to(device)<10.0,E_jp1_sum.to(device),torch.zeros(E_jp1_sum.shape).to(device)).to(device)
    E_jm1_sum=torch.where(E_jm1_sum.to(device)<10.0,E_jm1_sum.to(device),torch.zeros(E_jm1_sum.shape).to(device)).to(device)
    F_jp1_sum=torch.where(F_jp1_sum.to(device)<10.0,F_jp1_sum.to(device),torch.zeros(F_jp1_sum.shape).to(device)).to(device)
    F_jm1_sum=torch.where(F_jm1_sum.to(device)<10.0,F_jm1_sum.to(device),torch.zeros(F_jm1_sum.shape).to(device)).to(device)

    #print("E")
    #print(torch.max(E_jp1_sum))
    #print(torch.max(E_jm1_sum))
    #print("F")
    #print(torch.max(F_jp1_sum))
    #print(torch.max(F_jm1_sum))


    loss+=E_jp1_sum.sum()-E_jm1_sum.sum()
    loss+=F_jp1_sum.sum()-F_jm1_sum.sum()
    return loss**2/(input.shape[0]*input.shape[1]*input.shape[2]*input.shape[3])

(1,2,3,4)
print()

"""##  Prediction Loss"""

def PredictionLoss(fluid_prediction,fluid_gt):
    loss=fluid_prediction-fluid_gt
    loss=loss**2
    return torch.sum(loss)/(fluid_gt.shape[0]*fluid_gt.shape[1]*fluid_gt.shape[2]*fluid_gt.shape[3])

def loss_function(fluid_prediction,shock_prediction,fluid_gt,shock_gt,boundary):
    # shock label Loss
    #print(type(shock_prediction))
    #loss=nn.BCELoss(reduction="mean")(shock_prediction,shock_gt)

    # conservation Loss
    #loss=ConservationLoss(fluid_prediction,fluid_gt)
    loss=ConservationLoss(fluid_gt,fluid_prediction)
    #print(loss.item())

    # prediction los
    #loss=PredictionLoss(fluid_prediction,fluid_gt)
    #print(loss.item())

    # RH Loss
    #loss+=RHLoss(fluid_prediction,shock_prediction,fluid_gt,shock_gt,boundary)
    #loss=RHLoss(fluid_prediction,shock_prediction,fluid_gt,shock_gt,boundary)
    print(loss.item())
    
    
    return loss

shock_net=ShockNet().to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    shock_net= nn.DataParallel(shock_net, list(range(ngpu)))

# Apply the weights_init fun to randomly initialize all weights
#  to mean=0, stdev=0.2.
shock_net.apply(weights_init)

"""# Optimizer"""

optimizerS=optim.Adam(shock_net.parameters(),lr=lr,betas=(beta1,0.999))

"""# Training"""



# Lists to keep track of progress
epoch_list=[]
S_losses = []

shock_net.train()
print("Starting Training Loop...")
# For each epoch
#for epoch in tqdm(range(20)):
for epoch in tqdm(range(num_epochs)):
    epoch_loss=0.0
    epoch_list.append(epoch)

    # For each batch in the dataloader
    for i, data in enumerate(train_dataloader, 0):
        boundary,distance,input_labels,fluid,shock_labels=data
        #print(boundary[0,:,:])
        #plt.imshow(np.reshape(boundary[0,0:,:],(200,200)))
        #plt.show()

        ###########################
        # conver cpu to gpu
        boundary=boundary.to(device)
        distance=distance.to(device)
        input_labels=input_labels.to(device)
        fluid=fluid.to(device)
        shock_labels=shock_labels.to(device)

        # convert to Float
        boundary=boundary.type(torch.cuda.FloatTensor)
        distance=distance.type(torch.cuda.FloatTensor)
        input_labels=input_labels.type(torch.cuda.FloatTensor)
        fluid=fluid.type(torch.cuda.FloatTensor)
        shock_labels=shock_labels.type(torch.cuda.FloatTensor)

        ##########################
        ## Train with all-real batch
        shock_net.zero_grad()

        # predict fluid and shock labels
        fluid_prediction, shock_prediction= shock_net(distance,input_labels)

        # exclude the values inside the boundary 
        fluid_prediction, shock_prediction= fluid_prediction,shock_prediction

        # calculate loss on all batch
        loss=loss_function(fluid_prediction,shock_prediction,fluid,shock_labels,boundary)

        # add epoch loss
        epoch_loss=epoch_loss+loss.item()

        # calculate the gradients for this batch
        loss.backward()

        # update optimizer
        optimizerS.step()

    ###########################
    # post process 
    epoch_loss=epoch_loss/len(train_dataloader.dataset) 
    S_losses.append(epoch_loss)

    if epoch%3==1:
        print('[%d] loss: %.3f' % (epoch + 1, epoch_loss))
        plt.plot(epoch_list,S_losses,"r--")
        plt.legend("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    epoch_loss= 0.0
