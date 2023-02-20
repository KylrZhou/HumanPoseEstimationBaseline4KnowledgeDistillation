from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor,Normalize,RandomGrayscale,RandomHorizontalFlip,Compose,Resize
import torch.nn as nn
from torch.nn import Conv2d,BatchNorm2d,Sequential,MaxPool2d
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

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
    def __init__(self,downsample,out_channel):
        super(basic_block, self).__init__()
        if out_channel==64:
            in_channel=out_channel
        else:
            in_channel=out_channel/2
        self.res=Sequential()
        if downsample==True:
            self.conv1=Conv2d(in_channels=in_channel,out_channels=out_channel, kernel_size=3, stride=2,padding=1)
            self.res = Sequential(Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2,padding=1),
                                  BatchNorm2d(out_channel))
        else:
            self.conv1=Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.norm1=BatchNorm2d(out_channel)
        self.conv2=Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
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

class resnet(nn.Module):
    def __init__(self,block,layer_num):
        super(resnet, self).__init__()
        self.resize=Resize((224,224),interpolation=InterpolationMode.BILINEAR)
        self.stem_conv=Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.stem_norm=BatchNorm2d(64)
        self.stem_pooling=MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.stage1=self.makelayer(block,layer_num[0],64)
        self.stage2=self.makelayer(block,layer_num[1],128)
        self.stage3=self.makelayer(block,layer_num[2],256)
        self.stage4=self.makelayer(block,layer_num[3],512)
        #postprocess
    def makelayer(self,block,layer_num,process_channel):
        layer=Sequential()
        for i in range(layer_num):
            if process_channel!=64 and i==0:
                layer.append(block(downsample=True,out_channel=process_channel))
            else:
                layer.append(block(downsample=False,out_channel=process_channel))
        return layer

def ResNet18():
    #expandsion = 1
    Network = resnet(block=basic_block, layer_num=[2,2,2,2])
    return Network

def ResNet34():
    #expandsion = 1
    Network = resnet(block=basic_block, layer_num=[3,4,6,3])
    return Network
"""
def ResNet50():
    Network = resnet(block=bottleneck, layer_num=[3,4,6,3])
    return Network

def ResNet101():
    Network = resnet(block=bottleneck, layer_num=[3,4,23,3])
    return Network

def ResNet152():
    Network = resnet(block=bottleneck, layer_num=[3,8,36,3])
    return Network
"""
"""
Device = if("cuda avaible"): 'gpu0' else: 'cpu'
Network = ResNet34()
Network.to(Device)
Network.train()
"""

network=ResNet34()
print(network)