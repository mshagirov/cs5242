#!/usr/bin/env python
# coding: utf-8

# CS5242 Final Project : Model Training Notebook
# ======================================================= #
# > Transfer learning and fine-tuning ImageNet pre-trained models
#
# *Murat Shagirov*
# ======================================================= #

import numpy as np
from os import path

from nn import train_model # model training function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datautils import LoadTrainingData
from torch.utils.data import DataLoader
from torchvision import models, utils, transforms as T
# Optional, for plotting images
from datautils import BatchUnnorm, Unnorm


# check for CUDA device and set default dtype
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
dtype = torch.float32
print(f'device: {device}\ndtype: {dtype}')

# ======================================================= #
# ---------------------- Settings ----------------------- #
# ======================================================= #
# ----------- !!! Set These Before Running !!! ---------- #

# Paths to training dataset and labels (before Train/Val split)
train_csv = path.join('./datasets','train_label.csv')
train_data_path = path.join('./datasets','train_image','train_image')
# For saving models
model_fname = 'densenet121_ft_512px_v2' # file name for model
save_dir = './' # path to model folder

# Training hyperparam-s:
num_epochs = 3 # total number of epochs to train
bsize_train = 4 # training data batch sizes
bsize_val = 4 # validation data batch sizes
lr = 0.001 # learning rate
# More hyperparam-s below: "optimizer_ft": optimizer, "exp_lr_scheduler": lr policy

# ------- Transforms (Pre-processing Settings) --------- #
# Optional settings for plotting images:
unnorm = Unnorm() # unnormalize a single RGB image for ImageNet
unnormb = BatchUnnorm() # unnormalize batch of images for ImageNet
toPIL = T.ToPILImage() # optional, use it for plotting

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

# ----------- Initialize and split datasets ------------ #
np.random.seed(42) #seed np RNG for consistency
# Split the original training data into 85% / 15% train/val datasets
datasets = LoadTrainingData(train_csv, train_data_path, transform=transform,
                            split=True, train_percent=80, val_transform=val_transform)
print(f"Training dataset: {len(datasets['train'])} samples.",
      f"\nValidation dataset: {len(datasets['val'])} samples.")

# ======================================================= #

# For plotting
import matplotlib.pyplot as plt
# for plotting figures (report)
# import matplotlib
plt.style.use('ggplot')
# get_ipython().run_line_magic('matplotlib', 'inline')
# matplotlib.rcParams['figure.figsize'] = (15,5) # use larger for presentation
# matplotlib.rcParams['font.size']= 9 # use 14 for presentation

# ======================================================= #
# ---------------------- Finetuning --------------------- #
# ======================================================= #
# - fine tuning resnet18 seems faster, and validation set acc-y quickly reaches >85-90% after 5epochs
# - using resnet18's conv layer as feature extractor (freezing them) results in very slow training
#  (but no overfitting), both training and val-n set accuracies increase slowly (>80% after 5 epochs)
# - densenet121: so far 25*3 epochs-->97.4249%
# - resnext50_32x4d + fc: 25*1 epochs-->96.5665% (afterwards ValLoss converges to ~96%)
# ======================================================= #

# ------ Init, download weights, modify model ----------- #

# Download ImageNet pre-trained model from torchhub
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
best_model, curve_data  = train_model(model_ft, optimizer_ft, data_loaders, num_epochs=num_epochs,
                         loss_func=criterion, scheduler=exp_lr_scheduler, device=device, return_best=True)

# UNCOMMENT below for plotting loss and accuracy

# plt.figure(figsize=[20,8])
# t = np.arange(curve_data['total_epochs'])
# plt.subplot(121)
# plt.plot(t,curve_data['trainLosses'],label='Train')
# plt.plot(t,curve_data['valLosses'],label='Val')
# plt.title('Loss'); plt.legend()

# plt.subplot(122)
# plt.plot(t,curve_data['trainAccs'],label='Train')
# plt.plot(t,curve_data['valAccs'],label='Val')
# plt.title('Accuracy'); plt.legend()
# plt.show()

# ------------------- MODEL CHECKPOINT ------------------ #
# Save (current) best model:
torch.save(best_model.state_dict(), f'./{path.join(save_dir, model_fname)}.pkl')
# Save training/val losses and accuracies for plotting
torch.save(curve_data,f'./{path.join(save_dir, model_fname)}_plot.pkl')

# ======================================================= #
# ------------------- Further Training ------------------ #
# ======================================================= #

# # For changing LR-policy:
# # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3*len(datasets['train']), gamma=0.1)

# # continue training for 100 epochs:
# best_model, curve_data  = train_model(model_ft, optimizer_ft, data_loaders, num_epochs=100,
#                          loss_func=criterion, scheduler=exp_lr_scheduler, device=device, return_best=True)

# plt.figure(figsize=[20,8])
# t = np.arange(curve_data['total_epochs'])
# plt.subplot(121)
# plt.plot(t,curve_data['trainLosses'],label='Train')
# plt.plot(t,curve_data['valLosses'],label='Val')
# plt.title('Loss'); plt.legend()

# plt.subplot(122)
# plt.plot(t,curve_data['trainAccs'],label='Train')
# plt.plot(t,curve_data['valAccs'],label='Val')
# plt.title('Accuracy'); plt.legend()
# plt.show()

# ======================================================= #
# ------------------ Transfer Learning ------------------ #
# ======================================================= #
# My attempts on transfer learning
# fine-tuning results (above) were always better.
# Probable cause: I suspect, I did not disable BN layers in the convnet layers properly.
# Transfer learning performance on val dataset:
# - SGD moment0.9 lr=0.01: >90% (25 epochs)
# - SGD //-// lr= 0.01 step(20 epochs decay): 92% (100 epochs)
# ------------------------------------------------------- #

# num_epochs = 25

# bsize_train = 4 # batch sizes
# bsize_val = 4

# lr = 0.001 # learning rate

# # Download ImageNet pre-trained model from torchhub
# model_ft = models.resnet18(pretrained=True,progress=False)

# # for transfer learning freeze (disable grads for early layers)
# for param in model_ft.parameters():
#     param.requires_grad = False

# num_ftrs = model_ft.fc.in_features

# # size of each output sample: nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Sequential( nn.Linear(num_ftrs, num_ftrs),
#                           nn.ReLU(),
#                          nn.Linear(num_ftrs, 3))

# model_ft = model_ft.to(device)

# criterion = nn.CrossEntropyLoss()

# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=lr, momentum=0.9)
# # optimizer_ft = torch.optim.Adam(model_ft.fc.parameters(), lr=lr)

# # Decay LR by a factor of 0.1 every 7 epochs
# # exp_lr_scheduler = None
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5*len(datasets['train']), gamma=0.1)
# # exp_lr_scheduler = lr_scheduler.CyclicLR(optimizer_ft, 10**-9, 10**-4,
# #                                          step_size_up=5, step_size_down=10)

# # Prepare dataloaders
# data_loaders = {'train' : DataLoader(datasets['train'], batch_size=bsize_train, shuffle=True, num_workers=0),
#                 'val'   : DataLoader(datasets['val'],  batch_size=bsize_val, shuffle=False, num_workers=0)}
# best_model, curve_data  = train_model(model_ft, optimizer_ft, data_loaders, num_epochs=num_epochs,
#                          loss_func=criterion, scheduler=exp_lr_scheduler, device=device, return_best=True)

# plt.figure(figsize=[20,8])
# t = np.arange(curve_data['total_epochs'])
# plt.subplot(121)
# plt.plot(t,curve_data['trainLosses'],label='Train')
# plt.plot(t,curve_data['valLosses'],label='Val')
# plt.title('Loss'); plt.legend()

# plt.subplot(122)
# plt.plot(t,curve_data['trainAccs'],label='Train')
# plt.plot(t,curve_data['valAccs'],label='Val')
# plt.title('Accuracy'); plt.legend()

# plt.show()

# train_method = 'tr'
# model_name = 'resnet18'
# save_dir = '../../dataDIR/cs5242/'
# weights_path = path.join(save_dir, f'{model_name}_{train_method}_v1.pkl')
# torch.save(best_model.state_dict(), weights_path)
# torch.save(curve_data,f'./{model_name}_{train_method}_plot_v1.pkl')


# ======================================================= #
# ---------- Fine tuning for transfer learning: --------- #
# ======================================================= #
# Uncomment this section for fine-tuning (after pre-training the classifier/ FC layer)

# num_epochs = 50

# # re-enable grads for fine-tuning
# for param in model_ft.parameters():
#     param.requires_grad = True
# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00001, momentum=0.9)
# # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
# exp_lr_scheduler = lr_scheduler.CyclicLR(optimizer_ft, 10**-9, 10**-4,
#                                          step_size_up=5, step_size_down=10)

# best_model, curve_data  = train_model(model_ft, optimizer_ft, data_loaders, num_epochs=num_epochs,
#                          loss_func=criterion, scheduler=exp_lr_scheduler, device=device, return_best=True)

# plt.figure(figsize=[20,8])
# t = np.arange(curve_data['total_epochs'])
# plt.subplot(121)
# plt.plot(t,curve_data['trainLosses'],label='Train')
# plt.plot(t,curve_data['valLosses'],label='Val')
# plt.title('Loss'); plt.legend()

# plt.subplot(122)
# plt.plot(t,curve_data['trainAccs'],label='Train')
# plt.plot(t,curve_data['valAccs'],label='Val')
# plt.title('Accuracy'); plt.legend()

# plt.show()
# ======================================================= #
