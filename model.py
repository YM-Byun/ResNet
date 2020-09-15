import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {'resnet18': [2, 2, 2, 2], 'resnet34': [3, 4, 6, 3], 'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3], 'resnet152': [3, 8, 36, 3]}

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding, kernel_size=3, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                stride=1, padding=1, kernel_size=3, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.LeakyReLU(inplace=True)

        self.increase = None

        nn.init.constant_(self.bn2.weight, 0)
        

        if stride != 1 or in_channels != out_channels:
            self.increase = nn.Sequential(
                    nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels,
                        stride=stride),
                    nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.increase is not None:
            identity = self.increase(identity)

        x += identity
        x = self.relu(x)

        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                stride=stride, padding=1, kernel_size=3, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(out_channels*4)

        self.relu = nn.LeakyReLU(inplace=True)

        self.increase = None

        nn.init.constant_(self.bn2.weight, 0)
        

        if stride != 1 or in_channels != out_channels*4:
            self.increase = nn.Sequential(
                    nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels*4,
                        stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels*4))

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.increase is not None:
            identity = self.increase(identity)

        x += identity
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, model='resnet50'):
        super(ResNet, self).__init__()

        self.model = model
        self.inplanes = 64
        self.expansion = 1
        block = BasicBlock

        if (self.model == 'resnet50' or self.model == 'resnet101' or self.model == 'resnet152'):
            block = BottleneckBlock
            self.expansion = 4
        
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.inplanes, kernel_size=3,
                        padding=1, stride=1, bias=False),
                nn.BatchNorm2d(self.inplanes))

        self.layer2 = self.get_layer(block, channels=64, stride=1, n=cfg[self.model][0])
        self.layer3 = self.get_layer(block, channels=128, stride=2, n=cfg[self.model][1])
        self.layer4 = self.get_layer(block, channels=256, stride=2, n=cfg[self.model][2])
        self.layer5 = self.get_layer(block, channels=512, stride=2, n=cfg[self.model][3])

        self.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(512*self.expansion, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def get_layer(self, block, channels, stride, n):
        layers = [block(in_channels=self.inplanes, out_channels=channels,
            stride=stride)]

        self.inplanes = channels * self.expansion

        for i in range(n-1):
            layers.append(block(in_channels=self.inplanes, out_channels=channels))

        layers = nn.Sequential(*layers)

        return layers


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    dummy_data = torch.rand(4, 3, 32, 32)

    resnet = ResNet('resnet101')

    x = resnet(dummy_data)

    print (x.shape)
