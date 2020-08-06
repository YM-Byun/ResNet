import torch
import torchvision.transforms as transforms
import time
import sys
import torch.nn as nn
import argparse

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model import ResNet

batch_size =128
momentum=0.9
weight_decay = 0.0001
learning_rate = 0.1
epochs = 240
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

def get_parser():
    parser = argparse.ArgumentParser(description='ResNet')
    parser.add_argument('--gpu', type=int, default=-1,
            help='specific gpu num')

    parser.add_argument('--model', type=str, default='resnet18',
            help='ResNet model')

    args = parser.parse_args()

    return args

def main():
    parser = get_parser()

    global device

    if parser.gpu != -1:
        device = torch.device('cuda:' + str(parser.gpu))

    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    print ("\nLoading Cifar 10 Dataset...")

    train_dataset = CIFAR10(root='./dataset', train=True,
            download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4)

    val_dataset = CIFAR10(root='./dataset', train=False,
            download = True, transform=transform)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck')

    print ("Loaded Cifar 10!\n")

    print ("========================================\n")

    resnet = ResNet(parser.model)

    if is_cuda:
        resnet.to(device)

    global learning_rate

    optimizer = torch.optim.SGD(resnet.parameters(), lr=learning_rate, momentum=momentum,
            weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    best_acc = 0.0
    best_loss = 9.0

    if is_cuda:
        criterion, scheduler = criterion.to(device), scheduler.to(device)

    for epoch in range(epochs):
        train(train_loader, resnet, criterion, optimizer, epoch)

        print ("")

        acc, loss = validate(val_loader, resnet, criterion, epoch)

        scheduler.step(loss)

        is_best = False

        if best_acc == acc:
            if loss < best_loss:
                is_best = True
                best_loss = loss

        if best_acc < acc:
            is_best = True
            best_acc = acc

        if is_best:
            torch.save(resnet.state_dict(), "./weight/best_weight.pth")
            print (f"\nSave best model at acc: {acc:.4f},  loss: {loss:.4f}!")

        print ("\n========================================\n")

        torch.save(resnet.state_dict(), "./weight/lastest_weight.pth")

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    global device

    for i, data in enumerate(train_loader):
        inputs, label = data

        if is_cuda:
            inputs, label = inputs.to(device), label.to(device)

        optimizer.zero_grad()

        outputs =  model(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        acc1, acc5 = accuracy(outputs, label, topk=(1,5))

        if (i % 50 == 49) or (i == len(train_loader) - 1):
            print (f"Epoch [{epoch+1}/{epochs}] | Train iter [{i+1}/{len(train_loader)}] | acc1 = {acc1[0]:.3f} | acc5 = {acc5[0]:.3f} | loss = {(running_loss / float(i+1)):.5f} | lr = {get_lr(optimizer):.5f}")

def validate(val_loader, model, criterion, epoch):
    model.eval()
    running_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0

    global device

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, label = data

            if is_cuda:
                inputs, label = inputs.to(device), label.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, label)

            running_loss += loss.item()
            acc1, acc5 = accuracy(outputs, label, topk=(1,5))
            total_acc1 += acc1
            total_acc5 += acc5

    total_acc1 /= len(val_loader)
    total_acc5 /= len(val_loader)

    print (f"Epoch [{epoch+1}/{epochs}] | Validation | acc1 = {total_acc1[0]:.3f} | acc5 = {total_acc5[0]:.3f} | loss = {(running_loss / float(i)):.5f}")

    return total_acc1[0], (running_loss / float(i))

def get_lr(optimizer):
    lr = 0.0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    return lr

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


if __name__ == '__main__':
    main()
