# KITS19 participation with MIScnn

description blabla

## Prerequisites

Python version >= 3.6\
Pip\
git & git-lfs\

Install the MIScnn framework:
> git clone https://github.com/muellerdo/kits19.MIScnn.git
> pip install -r kits19.MIScnn/requirements.txt

Download the KITS19 data set
> git clone -b interpolated --single-branch https://github.com/neheller/kits19

## Training a model with MIScnn

Residual 3D U-Net training

> python MIScnn/train.py -i <kits19/data.interpolated>

asda

> cd kits19.MIScnn
> python MIScnn/train.py -i ../kits19/data
> cd ../

## Prediction of KITS19 test data with MIScnn

Residual 3D U-Net prediction

> python MIScnn/predict.py -i <kits19/data.interpolated>

asdasd

> cd kits19.MIScnn
> python MIScnn/predict.py -i ../kits19/data
> cd ../

## 3-fold Cross-Validation on KITS19 train set using MIScnn

plots

## Used hardware & software

Ubuntu 18.04\
Python, MIScnn, Keras, Tensorflow\
GPU

## Author

Dominik Müller\
Email: dominik.mueller@informatik.uni-augsburg.de\
IT-Infrastructure for Translational Medical Research\
University Augsburg\
Bavaria, Germany

## How to cite / More information

Dominik Müller and Frank Kramer. (2019)\
MIScnn: A Framework for Medical Image Segmentation with Convolutional Neural Networks and Deep Learning.

## License

This project is licensed under the MIT License.\
See the LICENSE.md file for license rights and limitations.
