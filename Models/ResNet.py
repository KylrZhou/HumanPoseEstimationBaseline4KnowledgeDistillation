from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Sequential, AdaptiveAvgPool2d, MaxPool2d

class BasicBlock(Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Sequential(Conv2d(in_channels = in_channels,
                                       out_channels = out_channels, 
                                       kernel_size = 3,
                                       stride = stride,
                                       padding = 1, bias = False),
                                BatchNorm2d(num_features = out_channels),
                                ReLU(inplace = True))
        self.conv2 = Sequential(Conv2d(in_channels = out_channels,
                                       out_channels = out_channels * BasicBlock.expansion,
                                       kernel_size = 3, 
                                       padding = 1, bias = False),
                                BatchNorm2d(num_features = out_channels * BasicBlock.expansion))
        self.shortcut = Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = Sequential(Conv2d(in_channels = in_channels,
                                              out_channels = out_channels * BasicBlock.expansion,
                                              kernel_size = 1,
                                              stride = stride, bias = False),
                                       BatchNorm2d(num_features = out_channels * BasicBlock.expansion))
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return ReLU(inplace = True)(x)

class BottleNeck(Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = Sequential(Conv2d(in_channels = in_channels,
                                       out_channels = out_channels, 
                                       kernel_size = 1, bias = False),
                                BatchNorm2d(num_features = out_channels),
                                ReLU(inplace = True))
        self.conv2 = Sequential(Conv2d(in_channels = out_channels,
                                       out_channels = out_channels,
                                       stride = stride,
                                       kernel_size = 3, 
                                       padding = 1, bias = False),
                                BatchNorm2d(num_features = out_channels),
                                ReLU(inplace = True))
        self.conv3 = Sequential(Conv2d(in_channels = out_channels,
                                       out_channels = out_channels * BottleNeck.expansion,
                                       kernel_size = 1, bias = False),
                                BatchNorm2d(num_features = out_channels * BottleNeck.expansion),)
        self.shortcut = Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = Sequential(Conv2d(in_channels = in_channels,
                                              out_channels = out_channels * BottleNeck.expansion, 
                                              stride = stride, 
                                              kernel_size = 1, bias = False),
                                       BatchNorm2d(num_features = out_channels * BottleNeck.expansion))
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return ReLU(inplace=True)(x)

class ResNet(Module):
    arch = {18 : ['BasicBlock', [2, 2, 2, 2]], 
            34 : ['BasicBlock', [3, 4, 6, 3]], 
            50 : ['BottleNeck', [3, 4, 6, 3]], 
            101 : ['BottleNeck', [3, 4, 23, 3]], 
            152 : ['BottleNeck', [3, 8, 36, 3]]}
    def __init__(self, total_layers):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.stem_conv = Sequential(Conv2d(in_channels = 3,
                                           out_channels = 64, 
                                           kernel_size = 7,
                                           stride = 2,
                                           padding = 3, bias = False),
                                    BatchNorm2d(num_features = 64),
                                    ReLU(inplace = True))
        self.max_pool = Sequential(MaxPool2d(kernel_size = 3,
                                             stride = 2,
                                             padding = 1),
                                   BatchNorm2d(num_features = 64),
                                   ReLU(inplace = True))
        self.Stage1 = self._make_layer(ResNet.arch[total_layers][0], 64, ResNet.arch[total_layers][1][0], 1)
        self.Stage2 = self._make_layer(ResNet.arch[total_layers][0], 128, ResNet.arch[total_layers][1][1], 2)
        self.Stage3 = self._make_layer(ResNet.arch[total_layers][0], 256, ResNet.arch[total_layers][1][2], 2)
        self.Stage4 = self._make_layer(ResNet.arch[total_layers][0], 512, ResNet.arch[total_layers][1][3], 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        block = eval(block)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return Sequential(*layers)

    def forward(self, x):
        output = self.stem_conv(x)
        output = self.max_pool(output)
        output = self.Stage1(output)
        output = self.Stage2(output)
        output = self.Stage3(output)
        output = self.Stage4(output)
        return output