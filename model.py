import torch
import torch.nn as nn
from torch.autograd import Variable

def conv(kernel_size, in_channels, out_channels, stride=1, padding=0):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding)

cfg = {'resnet152': [2, 7, 35, 2], 'resnet18': [1, 1, 1, 1]}

class Basicblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Basicblock, self).__init__()

        self.conv1 = conv(kernel_size=3, in_channels=in_channels, out_channels=out_channels,
                stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

        self.conv2 = conv(kernel_size=3, in_channels=out_channels, out_channels=out_channels,
                stride=stride, padding=1)

        self.relu = nn.LeakyReLU(inplace=True)

        self.increase_dimension = self.increase(in_channels=in_channels, out_channels=out_channels,
                stride=stride)

    def increase(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn(x)

        if x.shape != identity.shape:
            identity = self.increase_dimension(identity)

        x += identity

        x = self.relu(x)

        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = conv(kernel_size=1, in_channels=in_channels,
                out_channels=int(out_channels/4), stride=stride)

        self.bn = nn.BatchNorm2d(int(out_channels/4))

        self.conv2 = conv(kernel_size=3, in_channels=int(out_channels/4),
                out_channels=int(out_channels/4), stride=stride, padding=1)

        self.conv3 = conv(kernel_size=1, in_channels=int(out_channels/4),
                out_channels=out_channels, stride=stride)

        self.bn2 =nn.BatchNorm2d(out_channels)

        self.relu = nn.LeakyReLU(inplace=True)

        self.increase_dimension = self.increase(in_channels=in_channels, out_channels=out_channels,
                stride=stride)

    def increase(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        
        
        if identity.shape != x.shape:
            identity = self.increase_dimension(identity)

        x += identity
        

        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, resnet_type):
        super(ResNet, self).__init__()

        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, stride=1,
                    kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1))

        self.expansion = 1
        block = Basicblock

        if resnet_type == 'resnet152':
            self.expansion = 4
            block = Bottleneck

        self.layer2 = [block(in_channels=64, out_channels=64 * self.expansion)]
        for i in range(cfg[resnet_type][0]):
            self.layer2.append(block(in_channels=64*self.expansion, out_channels=64*self.expansion))
        self.layer2 = nn.Sequential(*self.layer2)

        self.layer3 = [block(in_channels=64*self.expansion, out_channels=128*self.expansion)]
        for i in range(cfg[resnet_type][1]):
            self.layer3.append(block(in_channels=128*self.expansion, out_channels=128*self.expansion))
        self.layer3 = nn.Sequential(*self.layer3)

        self.layer4 = [block(in_channels=128*self.expansion, out_channels=256*self.expansion)]
        for i in range(cfg[resnet_type][2]):
            self.layer4.append(block(in_channels=256*self.expansion, out_channels=256*self.expansion))
        self.layer4 = nn.Sequential(*self.layer4)

        self.layer5 = [block(in_channels=256*self.expansion, out_channels=512*self.expansion)]
        for i in range(cfg[resnet_type][3]):
            self.layer5.append(block(in_channels=512*self.expansion, out_channels=512*self.expansion))
        self.layer5 = nn.Sequential(*self.layer5)

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, 10)
        self.softmax = nn.Softmax(dim=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgPool(x)
        x = x.view(-1, 512*self.expansion)
        x = self.fc(x)
        x = self.softmax(x)

        return x


if __name__ == '__main__':
    dummy_data = torch.rand(1, 3, 32, 32)

    resnet = ResNet('resnet152')

    print ("ResNet 152\n")
    print (resnet)

    print ("\n=============================================\n")

    x = resnet(dummy_data)

    print (f"Result: {x.shape}")


    resnet = ResNet('resnet18')
    print ("Resnet 18\n")
    print (resnet)

    print ("\n=============================================\n")

    x = resnet(dummy_data)

    print (f"Result: {x.shape}")
