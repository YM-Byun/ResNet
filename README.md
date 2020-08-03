# Resnet 152
 
## 1. Trainning
 - Initalizing Conv layer using He initialization
 - Cifar 10 dataset, RandomCrop, RandomHorizontalFlip 
 - Increase learning rate from 0.1 to 0.0001

## 2. Result
 - At 46 epoch, acc: 81.25, loss: 1.997
 - still training...
 
## 3. Test
`` python3 test.py -i <IMAGE_PATH> ``

## Other models' acc
 1. 93.75% (Resnet 101 | https://github.com/kuangliu/pytorch-cifar)
 2. 93.57% (In originial paper)
