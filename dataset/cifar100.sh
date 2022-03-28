#!/bin/bash

# download and unzip dataset
wget https://github.com/hwany-j/cifar100_png/raw/main/cifar100_png.tar

tar -xvf cifar100_png.tar
mv cifar100_png cifar100
echo done