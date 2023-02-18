from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor,Normalize,RandomGrayscale,RandomHorizontalFlip,Compose
import torch.nn as nn
from torch.nn import Conv2d,BatchNorm2d,Sequential
import torch.nn.functional as F

Train_batch=64
Test_batch=1000

Train_Data=DataLoader(dataset=CIFAR10(
                                train=True,
                                root='./data/',
                                transform=Compose([RandomHorizontalFlip(),RandomGrayscale(),ToTensor(),Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
                                download=True),
                        batch_size=Train_batch,
                        shuffle=True)

Test_Data=DataLoader(dataset=CIFAR10(
                                train=False,
                                root='./data/',
                                transform=Compose([ToTensor(),Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
                                download=True),
                        batch_size=Test_batch,
                        shuffle=True)

class basic_block(nn.Module):
    def __init__(self,index,in_channel,out_channel):
        super(basic_block, self).__init__()
        self.res=Sequential()
        if index==1:
            self.conv1=Conv2d(in_channels=in_channel,out_channels=out_channel, kernel_size=3, stride=2,padding=1)
            self.res = Sequential(Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2,padding=1),
                                  BatchNorm2d(out_channel))
        else:
            self.conv1=Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=0)
        self.norm1=BatchNorm2d(out_channel)
        self.conv2=Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=0)
        self.norm2=BatchNorm2d(out_channel)
        self.norm3=BatchNorm2d(out_channel)
    def forward(self,x):
        residual=self.res(x)
        x=self.conv1(x)
        x=self.norm1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=self.norm2(x)
        x=F.relu(x)
        x+=residual
        x=self.norm3(x)
        x=F.relu(x)
        return x


