import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {'resnet18': [2, 2, 2, 2], 'resnet34': [3, 4, 6, 3]}

class Basicblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, kernel_size=3, increase=None):
        super(Basicblock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding, kernel_size=kernel_size, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                stride=1, padding=1, kernel_size=kernel_size, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

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


class ResNet(nn.Module):
    def __init__(self, model='resnet18'):
        super(ResNet, self).__init__()

        self.model = model
        self.inplanes = 64
        block = Basicblock
        
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.inplanes, kernel_size=3,
                        padding=1, stride=1),
                nn.BatchNorm2d(self.inplanes))

        self.layer2 = self.get_layer(block, channels=64, stride=1, n=cfg[self.model][0])
        self.layer3 = self.get_layer(block, channels=128, stride=2, n=cfg[self.model][1])
        self.layer4 = self.get_layer(block, channels=256, stride=2, n=cfg[self.model][2])
        self.layer5 = self.get_layer(block, channels=512, stride=2, n=cfg[self.model][3])

        self.fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(512, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def get_layer(self, block, channels, stride, n):
        increase = None

        if stride != 1 or self.inplanes != channels:
            increase = nn.Sequential(
                    nn.Conv2d(kernel_size=1, in_channels=self.inplanes, out_channels=channels,
                        stride=stride),
                    nn.BatchNorm2d(channels))


        layers = [block(in_channels=self.inplanes, out_channels=channels, stride=stride,
                increase=increase)]

        self.inplanes = channels

        for i in range(n-1):
            layers.append(block(in_channels=self.inplanes, out_channels=channels, increase=increase))

        layers = nn.Sequential(*layers)

        return layers


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    dummy_data = torch.rand(4, 3, 32, 32)

    resnet = ResNet()

    x = resnet(dummy_data)

    print (x.shape)
