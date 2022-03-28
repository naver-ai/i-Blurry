#!/bin/bash

# download and unzip dataset
wget https://github.com/hwany-j/cifar10_png/raw/main/cifar10_png.tar

tar -xvf cifar10_png.tar
mv cifar10_png cifar10
echo done