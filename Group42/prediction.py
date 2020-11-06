#!/usr/bin/env python
# coding: utf-8

# CS5242 Final Project : Prediction Script
# ===
# Before running please set param-s at "Set these before running" section below.
#
# *Murat Shagirov*

from os import path
import glob
import numpy as np
import pandas as pd
from skimage.io import imread

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datautils import LoadTrainingData
from torch.utils.data import DataLoader
from torchvision import models, utils, transforms as T

from datautils import BatchUnnorm, Unnorm

from nn import predict_check, predict_test # prediction function

# check for CUDA device and set default dtype
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
dtype = torch.float32
print(f'device: {device}\ndtype: {dtype}')


# ===================================================== #
# ----------- Set these before running ---------------- #
# ===================================================== #

# Set CHECK_VAL_PRED to True if you want to verify train/validation set performances
CHECK_VAL_PRED = True # : True/False

# location of training ID-label pairs
train_csv_path = './datasets/train_label.csv' # : str

# Path to directory with training set images
train_img_path = './datasets/train_image/train_image/' # : str

# Path to save to/load models from
models_path = './' # path : str
# model_fname = 'densenet121_ft_V1.pkl' # file name for desired model : str
model_fname = 'densenet121_ft_512px_v2.pkl' # file name for desired model : str

# !!! Path to directory with test dataset images !!!
test_path = '../datasets/test_image/test_image/' # : str

# ===================================================== #


# Transforms
img_size = 512 # Input image  size
# Training data transforms
transform = T.Compose([T.ToPILImage(),
                       T.RandomRotation((-3,3)),
                       T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                       T.RandomHorizontalFlip(),
                       T.ToTensor(),
                       T.ConvertImageDtype(dtype),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Test and Val data transforms
val_transform = T.Compose([T.ToPILImage(),
                           T.Resize(img_size),
                           T.ToTensor(),
                           T.ConvertImageDtype(dtype),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
test_transform = val_transform

# Loss function
loss_func = nn.CrossEntropyLoss()

# ===================================================== #
# ----------- Initiate and load desired model --------- #
# ===================================================== #
# Initiate default model (w/o weights) or download ImageNet pre-trained model from torchhub
model_ft = models.densenet121(pretrained=False,progress=False)
# change last FC layer
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, 3)

# Location for model's weights:
DENSENET121_path = path.join(models_path, model_fname)
model_ft.load_state_dict(torch.load(DENSENET121_path))
# set model mode for prediction:
model_ft.eval();


# ===================================================== #
# ------------ Train/ Val set performances ------------ #
# ===================================================== #
if CHECK_VAL_PRED:
    # Load and split original data into 80-20 Train-Val sets
    np.random.seed(42) #seed np RNG for consistency
    datasets = LoadTrainingData(train_csv_path, train_img_path, transform=transform,
                                split=True, train_percent=80, val_transform=val_transform)

    print(f"Training dataset: {len(datasets['train'])} samples.",
          f"\nValidation dataset: {len(datasets['val'])} samples.")

    # Batch sizes
    bsize_train = 4 # Training
    bsize_val = 4 # Val and Test

    # Prepare dataloaders
    # Set SHUFFLE==FALSE
    data_loaders = {'train' : DataLoader(datasets['train'], batch_size=bsize_train,
                     shuffle=False, num_workers=0),
                    'val'   : DataLoader(datasets['val'],  batch_size=bsize_val,
                     shuffle=False, num_workers=0)}

    # Run prediction on training and validation datasets:
    losses, accuracies, pred_labels = predict_check(data_loaders, model_ft, device=device)
else:
    print('Skipping train/validation set performances.')


# ===================================================== #
# --------------- Test set predictions ---------------- #
# ===================================================== #
pred_labels = predict_test(test_path, model_ft, test_transform,
                           batch_size=4, device=device)

# Write to csv file:
preds_df = pd.DataFrame(data=pred_labels).sort_values('ID') # to pd.DataFrame
preds_df.set_index('ID').to_csv('./test_submission.csv',sep=',')


# END
