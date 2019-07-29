# KITS19 participation with MIScnn

We participated at the Kidney Tumor Segmentation Challenge 2019 (KITS19) with our newly developed framework for medical image segmentation. MIScnn: A Framework for Medical Image Segmentation with Convolutional Neural Networks and Deep Learning.

The aim of MIScnn is to provide an intuitive API allowing fast building of medical
image segmentation pipelines including data I/O, preprocessing, data augmentation, patch-wise analysis, metrics, a library with state-of-the-art
deep learning models and model utilization like training, prediction as well as fully automatic evaluation (e.g. cross-validation).
Even so, high configurability and multiple open interfaces allow full pipeline customization. MIScnn is based on Keras with Tensorflow as backend.\
More information about MIScnn can be found in the publication or on the Git repository: https://github.com/frankkramer-lab/MIScnn

The task of the KITS19 challenge was to compute a semantic segmentation of arterial phase abdominal CT scans from 300 kidney cancer patients. Each pixel had to be labeled into one of three classes: Background, kidney or tumor. The original scans have an image resolution of 512x512 and on average 216 slices (highest slice number is 1059).

## MIScnn configuration for the KITS19: 3D Residual U-Net model

For the deep learning model architecture, we selected a 3D Residual U-Net model for in-depth image processing
over all three axes. We selected a patch-wise analysis with 48x128x128 patches without pixel value normalization in the preprocessing,
but full data augmentation (translation, rotation and flipping) with 12x32x32 patch overlaps.
During training, blank patches were skipped and overlapping patches are created for prediction. The whole KITS19 interpolated data set were used for fitting. The training was performed using the Tversky loss for 20 epochs with a learning rate of 1E-4, batch shuffling and a batch size of 15.

## Prerequisites

Python version >= 3.6\
git & git-lfs

Install the MIScnn framework:
```sh
git clone https://github.com/muellerdo/kits19.MIScnn.git
pip install -r kits19.MIScnn/requirements.txt
```

Download the interpolated KITS19 data set:
```sh
git clone -b interpolated --single-branch https://github.com/neheller/kits19
```

Create a link to the data set and go into the kits19.MIScnn directory:
```sh
ln -sr kits19/data kits19.MIScnn/data
cd kits19.MIScnn
```

## Training a model with MIScnn

In order to train the 3D Residual U-Net model on the KITS19 data set run the kits19_train.py script on the data directory link.\
The Python script contains the used configurations for MIScnn and calls the MIScnn training functions with these configurations.

```sh
python kits19_train.py -i data
```

The training process takes up to 37 hours using two Nvidia Quadrop P6000 with in total 30 GB memory.\
The resulting fitted model was saved under the sub directory "model/".\
For the KITS19 prediction, the training can be skipped by just using the already fitted model file.

## Prediction of KITS19 test data with MIScnn

In order to make segmentation predictions for the KITS19 test data set with a fitted 3D Residual U-Net model, run the kits19_predict.py script on the data directory link.\
The Python script contains the used configurations for MIScnn and calls the MIScnn prediction functions with these configurations.

```sh
python kits19_predict.py -i data
```

The resulting predictions were saved under the sub directory "predictions/".

## 10% Split-Validation on KITS19 train set with MIScnn

The complete KITS19 test data set was split into a 90% training and 10% testing set. Instead of 20 epochs (which was used for the submission) only 15 epochs were used for the validation.\
In order to run the automatic 10% Split-Validation for the KITS19 test data set with a fitted 3D Residual U-Net model, run the kits19_evaluate.py script on the data directory link.\
The Python script contains the used configurations for MIScnn and calls the MIScnn evaluation functions with these configurations.

```sh
python kits19_evaluate.py -i data
```

The resulting evaluation figures and scores were saved under the sub directory "evaluation/".

![evaluation plots](evaluation/multiplot.png)

![example gif](evaluation/visualization.case_00141.gif)

## Used hardware & software

Ubuntu 18.04\
Python, MIScnn, Keras, Tensorflow\
GPU: 2x Nvidia Quadro P6000 with 24GB memory

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

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3.\
See the LICENSE.md file for license rights and limitations.
