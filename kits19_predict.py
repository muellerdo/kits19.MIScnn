#!/usr/bin/env python
# -*- coding: utf-8 -*-

#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import sys
import argparse
import os
import numpy as np
# Internal libraries/scripts
import MIScnn.neural_network as MIScnn_NN
import MIScnn.evaluation as MIScnn_CV
from MIScnn.data_io import save_prediction

#-----------------------------------------------------#
#                  Parse command line                 #
#-----------------------------------------------------#
# Implement a modified ArgumentParser from the argparse package
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message + "\n")
        self.print_help()
        sys.exit(2)
# Initialize the modifed argument parser
parser = MyParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                add_help=False, description=
    """
Description...

Author: Dominink Müller
Email: dominik.mueller@informatik.uni-augsburg.de
Chair: IT-Infrastructure for Translational Medical Research- University Augsburg (Germany)
""")
# Add arguments for mutally exclusive required group
required_group = parser.add_argument_group(title='Required arguments')
required_group.add_argument('-i', '--input', type=str, action='store',
                required=True, dest='args_input', help='Path to data directory')
# Add arguments for optional group
optional_group = parser.add_argument_group(title='Optional arguments')
optional_group.add_argument('-v', '--verbose', action='store_true',
                default=False, dest='args_verbose',
                help="Print all informations and warnings")
optional_group.add_argument('-h', '--help', action="help",
                help="Show this help message and exit")
# Parse arguments
args = parser.parse_args()

#-----------------------------------------------------#
#                   Configurations                    #
#-----------------------------------------------------#
config = dict()
# Dataset
config["cases"] = list(range(210,300))
config["data_path"] = args.args_input           # Path to the kits19 data dir
config["model_path"] = "model"                  # Path to the model data dir
config["output_path"] = "predictions"           # Path to the predictions directory
config["evaluation_path"] = "evaluation"        # Path to the evaluation directory
# GPU Architecture
config["gpu_number"] = 2                        # Number of GPUs (if > 2 = multi GPU)
# Neural Network Architecture
config["input_shape"] = (None, 128, 128, 1)     # Neural Network input shape
config["patch_size"] = (48, 128, 128)           # Patch shape/size
config["classes"] = 3                           # Number of output classes
config["batch_size"] = 10                       # Number of patches in on step
# Training
config["epochs"] = 20                           # Number of epochs for training
config["max_queue_size"] = 3                    # Number of preprocessed batches
config["learninig_rate"] = 0.0001               # Learninig rate for the training
config["shuffle"] = True                        # Shuffle batches for training
# Data Augmentation
config["overlap"] = (12, 32, 32)                # Overlap in (x,y,z)-axis
config["skip_blanks"] = True                    # Skip patches with only background
config["scale_input_values"] = False            # Scale volume values to [0,1]
config["rotation"] = True                       # Rotate patches in 90/180/270°
config["flipping"] = True                       # Reflect/Flip patches
config["flip_axis"] = (3)                       # Define the flipping axes (x,y,z <-> 1,2,3)
# Prediction
config["pred_overlap"] = True                   # Usage of overlapping patches in prediction
# Evaluation
config["n_folds"] = 3                           # Number of folds for cross-validation
config["per_split"] = 0.095                     # Percentage of Testing Set for split-validation
config["n_loo"] = 3                             # Number of cycles for leave-one-out
config["visualize"] = True                      # Print out slice images for visual evaluation
config["class_freq"] = False                    # Calculate the class frequencies for each slice


#-----------------------------------------------------#
#          GPU Management for shared hardware         #
#-----------------------------------------------------#
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#-----------------------------------------------------#
#                    Runner code                      #
#-----------------------------------------------------#
# Output the configurations
print(config)

# Create the 3D Residual U-Net
cnn_model = MIScnn_NN.NeuralNetwork(config)

# Load the already fitted model from file
cnn_model.load("residual_unet")

# Predict kits19 test set with the fitted 3D Residual U-Net model
cnn_model.predict(config["cases"])

# Resize enlarged 268 ct sample to original size back
pred = load_prediction_nii(268, config["output_path"]).get_data()
vol = load_volume_nii(268, config["data_path"]).get_data()
pred = np.resize(pred, vol.shape)
save_prediction(pred, 268, config["output_path"])
