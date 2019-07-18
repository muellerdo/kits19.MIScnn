# KITS19 participation with MIScnn

We participated at the Kidney Tumor Segmentation Challenge 2019 with our newly developed framework for medical image segmentation. MIScnn: A Framework for Medical Image Segmentation with Convolutional Neural Networks and Deep Learning.

The aim of MIScnn is to provide an intuitive API allowing fast building of medical
image segmentation pipelines including data I/O, preprocessing, data augmentation, patch-wise analysis, metrics, a library with state-of-
the-art deep learning models and model utilization like training, prediction as well as fully automatic evaluation (e.g. cross-validation).
Even so, high configurability and multiple open interfaces allow full pipeline customization.

## MIScnn configuration for the KITS19

For the deep learning model architecture, we selected a 3D Residual U-Net model for in-depth image processing
over all three axes. We selected a patch-wise analysis with 48x128x128 patches without pixel value normalization in the preprocessing,
but full data augmentation (translation, rotation and flipping) with 12x32x32 patch overlaps.
During training, blank patches were skipped and no overlapping patches are created for prediction. The
whole KITS19 interpolated data set were used for fitting. The training was performed using the Tversky loss (alpha&beta == 0.5) for #40#
epochs with a learning rate of 1E-4, batch shuffling and a batch size of #X#. #24??

## Prerequisites

Python version >= 3.6\
Pip\
git & git-lfs

Install the MIScnn framework:
> git clone https://github.com/muellerdo/kits19.MIScnn.git\
> pip install -r kits19.MIScnn/requirements.txt

Download the interpolated KITS19 data set and create a link:
> git clone -b interpolated --single-branch https://github.com/neheller/kits19\
> ln -sr kits19/data kits19.MIScnn/data
> cd kits19.MIScnn

## Training a model with MIScnn

In order to train the 3D Residual U-Net model on the KITS19 data set run the MIScnn/train.py script on the data directory link.\
The Python script contains the used configurations for MIScnn and calls the MIScnn training functions with these configurations.

> python MIScnn/train.py -i data

## Prediction of KITS19 test data with MIScnn

In order to make segmentation predictions for the KITS19 test data set with a fitted 3D Residual U-Net model, run the MIScnn/predict.py script on the data directory link.\
The Python script contains the used configurations for MIScnn and calls the MIScnn prediction functions with these configurations.

> python MIScnn/predict.py -i data

## 3-fold Cross-Validation on KITS19 train set with MIScnn

plots

## Used hardware & software

Ubuntu 18.04\
Python, MIScnn, Keras, Tensorflow\
2x Nvidia Quadro P6000 with 24GB memory

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
