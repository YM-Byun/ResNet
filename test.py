import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import os

from model import ResNet
from torch.autograd import Variable

is_cuda = torch.cuda.is_available()

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck')

def main():
    parser = get_argparser()

    img_path = parser.i

    if not os.path.isfile(img_path):
        print ("No input file")
        return

    img = load_img(img_path)

    print ("\nLoading ResNet 18 weight...")

    resnet = ResNet(parser.model)
    resnet.load_state_dict(torch.load(parser.weight), strict=False)

    if is_cuda:
        resnet.cuda()

    print ("\nLoaded ResNet 18 network!\n")

    print ("==================================\n")

    classify(resnet, img)

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', type=str,
            default='./test_img/deer.jpg', help='input image path')

    parser.add_argument('--model', type=str,
            default='resnet18')

    parser.add_argument('--weight', type=str,
            default='./weight/93_12_resnet18.pth')

    args = parser.parse_args()

    return args


def load_img(path):
    transform = transforms.Compose([
            transforms.Resize((32, 32)),
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


if __name__ == "__main__":
    main()
