import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = self.conv(kernel_size=1, in_channels=in_channels,
                out_channels=int(out_channels/4), stride=stride)

        self.bn = nn.BatchNorm2d(int(out_channels/4))

        self.conv2 = self.conv(kernel_size=3, in_channels=int(out_channels/4),
                out_channels=int(out_channels/4), stride=stride, padding=1)

        self.conv3 = self.conv(kernel_size=1, in_channels=int(out_channels/4),
                out_channels=out_channels, stride=stride)

        self.bn2 =nn.BatchNorm2d(out_channels)

        self.relu = nn.LeakyReLU(inplace=True)

    def conv(self, kernel_size, in_channels, out_channels, stride=1, padding=0):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)

    def down_sampling(self, x, i):
        i = nn.Conv2d(kernel_size=1, in_channels=i.shape[1],
                out_channels=x.shape[1], stride=1)(i)
        i = nn.BatchNorm2d(x.shape[1])(i)
        
        return i

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn(x)

        x = self.conv2(x)
        x = self.bn(x)

        x = self.conv3(x)
        x = self.bn2(x)

        if x.shape != identity.shape:
            identity = self.down_sampling(x, identity)

        x = x + identity

        return x

class ResNet_152(nn.Module):
    def __init__(self):
        super(ResNet_152, self).__init__()

        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, stride=2,
                    kernel_size=7, padding=3),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer2 = [Bottleneck(in_channels=64, out_channels=256)]
        for i in range(2):
            self.layer2.append(Bottleneck(in_channels=256, out_channels=256))
        self.layer2 = nn.Sequential(*self.layer2)

        self.layer3 = [Bottleneck(in_channels=256, out_channels=512)]
        for i in range(7):
            self.layer3.append(Bottleneck(in_channels=512, out_channels=512))
        self.layer3 = nn.Sequential(*self.layer3)

        self.layer4 = [Bottleneck(in_channels=512, out_channels=1024)]
        for i in range(35):
            self.layer4.append(Bottleneck(in_channels=1024, out_channels=1024))
        self.layer4 = nn.Sequential(*self.layer4)

        self.layer5 = [Bottleneck(in_channels=1024, out_channels=2048)]
        for i in range(2):
            self.layer5.append(Bottleneck(in_channels=2048, out_channels=2048))
        self.layer5 = nn.Sequential(*self.layer5)

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, 10)
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
        x = x.view(-1, 512*4)
        x = self.fc(x)
        x = self.softmax(x)

        return x


if __name__ == '__main__':
    dummy_data = torch.rand(1, 3, 32, 32)

    resnet = ResNet_152()

    print ("ResNet 152\n")
    print (resnet)

    print ("\n=============================================\n")

    x = resnet(dummy_data)

    print (f"Result: {x.shape}")
