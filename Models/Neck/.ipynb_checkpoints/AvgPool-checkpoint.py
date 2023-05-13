from torch.nn import Module, AdaptiveAvgPool2d

class GlobalAvgPool(Module):
    def __init__(self, output_channels):
        super(GlobalAvgPool, self).__init__()
        self.Pooling = AdaptiveAvgPool2d(output_size = (1,1))
        self.output_channels = output_channels
        
    def forward(self, x):
        x = self.Pooling(x)
        x = x.view(-1, self.output_channels)
        return x