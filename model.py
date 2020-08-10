import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {'resnet20': [2, 2, 2, 2], 'resnet34': [3, 4, 6, 3]}

class Basicblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, kernel_size=3, increase=None):
        super(Basicblock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding, kernel_size=kernel_size)

        self.bn = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                stride=1, padding=1, kernel_size=kernel_size)

        self.relu = nn.ReLU(inplace=True)

        self.increase = increase


    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn(x)

        if self.increase is not None:
            identity = self.increase(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, model='resnet20'):
        super(ResNet, self).__init__()

        self.model = model
        block = Basicblock
        
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                        padding=1, stride=1),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer2 = self.get_layer(block, channels=64, stride=1, n=cfg[self.model][0])
        self.layer3 = self.get_layer(block, channels=128, stride=2, n=cfg[self.model][1])
        self.layer4 = self.get_layer(block, channels=256, stride=2, n=cfg[self.model][2])
        self.layer5 = self.get_layer(block, channels=512, stride=2, n=cfg[self.model][3])

        self.avgPool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 10))


    def get_layer(self, block, channels, stride, n):
        increase = None
        expansion = 1

        if stride == 2:
            increase = nn.Sequential(
                    nn.Conv2d(kernel_size=1, in_channels=channels//2, out_channels=channels,
                        stride=2),
                    nn.BatchNorm2d(channels))

            expansion = 2

        layers = [block(in_channels=channels//expansion, out_channels=channels, stride=stride,
                increase=increase)]

        for i in range(n-1):
            layers.append(block(in_channels=channels, out_channels=channels))

        layers = nn.Sequential(*layers)

        return layers


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    dummy_data = torch.rand(4, 3, 32, 32)

    resnet = ResNet()

    x = resnet(dummy_data)

    print (x.shape)
