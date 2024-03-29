from torch.nn import Module, Sequential, Linear
import sys

sys.path.append('../..')

class FullyConnectedHead(Module):
    def __init__(self, prog_channels):
        super(FullyConnectedHead, self).__init__()
        fc = []
        for i in range(len(prog_channels)-1):
            fc.append(Linear(in_features = prog_channels[i], out_features = prog_channels[i+1]))
        self.fc = Sequential(*fc)
    
    def forward(self, x):
        x = self.fc(x)
        return x