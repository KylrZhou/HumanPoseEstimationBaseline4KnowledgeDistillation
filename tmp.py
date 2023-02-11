import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

train_data =DataLoader(dataset= torchvision.datasets.MNIST(
                            train=True,
                            root="./data/",
                            download=True,
                            transform= torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,),(0.3081,)),
                            ])
                        ),
                        batch_size=64,
                        shuffle=True,
)

test_data=DataLoader(dataset=torchvision.datasets.MNIST(
                        train=False,
                        root="./data/",
                        download=True,
                        transform= torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,),(0.3081,)),
                        ])
                    ),
                    batch_size=1000,
                    shuffle= True
)

'''
data=enumerate(test_data)
data=next(data)
fig=plt.figure()
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(data[1][0][i][0],cmap="gray",interpolation="none")
    plt.title(f"GTis{data[1][1][i]}")

plt.show()
'''

randomseed=np.random.uniform()
torch.manual_seed(randomseed)
epochs=3
batch_size_train=64
batch_size_test=1000
learning_rate=0.01
momentum=0.5
log_interval=10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(1,10,5)
        self.conv2=nn.Conv2d(10,20,5)
        self.conv2_dropout=nn.Dropout2d()
        self.fc1=nn.Linear(320,50)
        self.fc2=nn.Linear(50,10)

    def forward(self,x):
        x=self.conv1(x)
        x=F.max_pool2d(x,2)
        x=F.relu(x)
        x=self.conv2(x)
        x=self.conv2_dropout(x)
        x=F.max_pool2d(x,2)
        x=F.relu(x)
        x=x.view(-1,320)
        x=self.fc1(x)
        x=F.relu(x)
        x=F.dropout(x,training=self.training)
        x=self.fc2(x)
        x=F.log_softmax(x,dim=-1)
        return x

network=Net()
optimizer=optim.SGD(network.parameters(),lr=learning_rate,momentum=momentum)




