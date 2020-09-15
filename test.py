import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import os

from cifar_model import ResNet
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

device = torch.device('cuda')

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck')

def main():
    global device

    print ("\nLoading ResNet weight...")

    model = 'resnet110'

    resnet = ResNet(model)
    resnet.load_state_dict(torch.load('./weight/93_68_' + model +'.pth'))

    resnet.to(device)

    print ("\nLoaded ResNet network!\n")

    print ("==================================\n")

    #classify(resnet, img)

    print ("Loading dataset...\n\n")

    cifar10_dataset = CIFAR10(root='./dataset', train=True,
            download=True)

    mean, std = get_mean_std(cifar10_dataset)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    test_dataset = CIFAR10(root='./dataset', train=False, download = False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=8)

    test(test_loader, resnet)
        
def get_mean_std(dataset):
    mean = dataset.data.mean(axis=(0,1,2,)) / 255
    std = dataset.data.std(axis=(0,1,2,)) / 255

    return mean, std

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', type=str,
            default='./test_img/deer.jpg', help='input image path')

    args = parser.parse_args()

    return args


def load_img(path):
    transform = transforms.Compose([transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])

    image = Image.open(path)
    image = transform(image)
    image = image.unsqueeze(0)
    image = Variable(image)

    if is_cuda:
        image = image.cuda()

    return image

def classify(model, inputs):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        outputs = model(inputs)

    val, indices = torch.topk(outputs, 5)

    print ("Result:")

    for i in range(5):
        print (f'{i+1}.')
        print (f'\tClass: {classes[indices[0][i].item()]}')
        print (f'\tAcc: {val[0][i].item():.5f}\n')

def test(test_loader, model):
    global device

    model.eval()
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    total_acc1 = 0.0
    total_acc5 = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, label = data

            inputs, label = inputs.to(device), label.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, label)

            running_loss += loss.item()
            acc1, acc5 = accuracy(outputs, label, topk=(1,5))

            total_acc1 += acc1
            total_acc5 += acc5

        total_acc1 /= len(test_loader)
        total_acc5 /= len(test_loader)

    print (f"Test | acc1 = {total_acc1[0]:.3f} | acc5 = {total_acc5[0]:.3f} | loss = {running_loss:.5f}")

    return total_acc1[0], running_loss

def accuracy(output, label, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = label.size(0)

        _,pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1,-1).expand_as(pred))

        res = []

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))

        return res

if __name__ == "__main__":
    main()
