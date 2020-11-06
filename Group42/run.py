#!/usr/bin/env python
# coding: utf-8

# CS5242 Final Project : Training and Prediction Script
# ===
# *Murat Shagirov*

import argparse
from os import path
import sys
sys.path.insert(1,path.join('.','code'))

import numpy as np
import glob
import pandas as pd
from skimage.io import imread

from datautils import LoadTrainingData
from nn import train_model # model training function
from nn import predict_check, predict_test # prediction function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, utils, transforms as T


parser = argparse.ArgumentParser()
parser.add_argument('train_data', nargs=1)
parser.add_argument('test_data', nargs=1)

args = parser.parse_args()

print(f'\nTraining data path:{args.train_data[0]}')
print(f'Test data path:{args.test_data[0]}\n')

# Set CHECK_VAL_PRED to True if you want to verify train/validation set performances
CHECK_VAL_PRED = True # : True/False

# ---------------------- Paths ----------------------- #
# Paths to training dataset and labels (before Train/Val split)
train_root = args.train_data[0]
train_csv = path.join(train_root,'train_label.csv')
train_data_path = path.join(train_root,'train_images')
test_path = path.join(args.test_data[0],'test_images')
# For saving models
model_fname = 'densenet121_ft_512px_v2' # file name for model
model_save_path = path.join('.','ckp',model_fname) # path to model folder


# ---------------------- Set Device ------------------ #
# check for CUDA device and set default dtype
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
dtype = torch.float32
print(f'device: {device}\ndtype: {dtype}')

# ---------------------------------------------------- #
# -------------- Training (Fine-tuning) -------------- #
# ---------------------------------------------------- #
num_epochs = 200 # total number of epochs to train
bsize_train = 4 # training data batch sizes
bsize_val = 4 # validation data batch sizes
lr = 0.001 # learning rate
# More hyperparam-s below: "optimizer_ft": optimizer, "exp_lr_scheduler": lr policy
# Pre-processing settings (defaults for ImageNet dataset)
img_size = 512
transform = T.Compose([T.ToPILImage(),
                       T.RandomRotation((-3,3)),
                       T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                       T.RandomHorizontalFlip(),
                       T.ToTensor(),
                       T.ConvertImageDtype(dtype),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
val_transform = T.Compose([T.ToPILImage(),
                           T.Resize(img_size),
                           T.ToTensor(),
                           T.ConvertImageDtype(dtype),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transform = val_transform

# ----------- Initialize and split datasets ------------ #
np.random.seed(42) #seed np RNG for consistency
# Split the original training data into 85% / 15% train/val datasets
datasets = LoadTrainingData(train_csv, train_data_path, transform=transform,
                            split=True, train_percent=80, val_transform=val_transform)
print(f"Training dataset: {len(datasets['train'])} samples.",
      f"\nValidation dataset: {len(datasets['val'])} samples.")

# ------ Init, download weights, modify model ----------- #

# Download ImageNet pre-trained model from torchhub
print('\nDownloading pre-trained denseNet121 ...')
model_ft = models.densenet121(pretrained=True,progress=False)

# Init last classifier (FC) layer:
# num_ftrs = model_ft.fc.in_features
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, 3)

# ------------------- Model Training -------------------- #

model_ft = model_ft.to(device)

# Finetune all parameters
criterion = nn.CrossEntropyLoss() # loss function (+softmax)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

# LR Schedules
# for no policy, use : exp_lr_scheduler = None
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5*len(datasets['train']), gamma=0.1)

# Prepare dataloaders
data_loaders = {'train' : DataLoader(datasets['train'], batch_size=bsize_train, shuffle=True, num_workers=0),
                'val'   : DataLoader(datasets['val'],  batch_size=bsize_val, shuffle=False, num_workers=0)}

# for submission num_epochs > 60 epochs
print('\nTraining the model:')
best_model, curve_data  = train_model(model_ft, optimizer_ft, data_loaders, num_epochs=num_epochs,
                         loss_func=criterion, scheduler=exp_lr_scheduler, device=device, return_best=True)

# ------------------- MODEL CHECKPOINT ------------------ #
# Save (current) best model:
# torch.save(best_model.state_dict(), f'./{model_save_path}.pkl')
# # Save training/val losses and accuracies for plotting
# torch.save(curve_data,f'./{model_save_path}_plot.pkl')


# ------------------- Evaluate Model ------------------- #
model_ft=best_model
# set model mode for prediction:
model_ft.eval();

if CHECK_VAL_PRED:
    print('\nModel Evaluation\n'+'-'*10)
    # Run prediction on training and validation datasets:
    losses, accuracies, pred_labels = predict_check(data_loaders, model_ft, device=device)
    print('~'*10+'\n')
else:
    print('\nSkipping train/validation set performances.')

# --------------- Test set predictions ---------------- #
pred_labels = predict_test(test_path, model_ft, test_transform, batch_size=4, device=device)

# Write to csv file:
print('Saving test.csv')
preds_df = pd.DataFrame(data=pred_labels).sort_values('ID') # to pd.DataFrame
preds_df.set_index('ID').to_csv(path.join('.','test_result.csv'),sep=',')

