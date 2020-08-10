# Resnet
 
## 1. Trainning
 - Initalizing Conv layer using He initialization
 - Cifar 10 dataset, RandomCrop, RandomHorizontalFlip 
 - Increase learning rate from 0.1 to 0.0001
 - Increase dimension using 1x1 conv layer

## 2. Result
 - ResNet18 | acc: 94.06% loss: 0.1880
 - still training...
 
## 3. Test
`` python3 test.py -i <IMAGE_PATH> ``

## Other models' acc
 1. 93.75% (Resnet 101 | https://github.com/kuangliu/pytorch-cifar)
 2. 93.57% (In originial paper)
