import torch
import numpy as np
import pandas as pd
from skimage.io import imread
from os import path
from torch.utils.data import Dataset, DataLoader

class DatasetFromArray(Dataset):
    '''CS5242 Final Project Dataset (Images look like lung X-ray images).'''
    def __init__(self, label_id_pairs, image_dir, transform=None):
        '''
        Arg-s:
        - label_id_pairs (Nx2 array): values from "*.csv" file that contains [ID, label] pairs.
        - image_dir (string): full path to the directory with input images named with "ID.png" pattern.
        - transform (callable): optional transformations on input images {default: None}.
        '''
        super(DatasetFromArray).__init__()
        self.label_id_pairs = label_id_pairs
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        '''Returns dataset size.'''
        return self.label_id_pairs.shape[0]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_fname = path.join(self.image_dir, f'{self.label_id_pairs[idx,0]}.png')
        x = imread(img_fname)
        y = self.label_id_pairs[idx,1]

        if self.transform:
            x = self.transform(x)

        return {'image':x, 'label': y}

def LoadTrainingData(csv_file, image_dir, transform=None, split=True, train_percent=85, val_transform=None):
    '''
    Load and optionally split the training dataset, returns `torch.utils.data.Dataset` objects.

    Arg-s:
    - csv_file (string): full path to "*.csv" file that contains [ID, label] pairs.
    - image_dir (string): full path to the directory with input images named with "ID.png" pattern.
    - transform (callable): optional transformations on training dataset images {default: None}.
    Validation dataset does not have transformations.
    - split: True/False, randomly split to training and validation datasets.
    - train_percent: fraction of training data, train_percent/100 {default: 85}
                     `train_size = (number_of_samples * train_percent)//100`.
    - val_transform (callable): transform for validation set (e.g. Convert to tensor, normalise etc.).
    '''
    label_id_pairs = pd.read_csv(csv_file).values
    datasets = {}
    if split:
        IDs_shuffled = np.random.permutation(label_id_pairs.shape[0])
        train_size = label_id_pairs.shape[0]*train_percent//100
        # val_size   = label_id_pairs.shape[0] - train_size

        # split dataset
        train_label_id_pairs = label_id_pairs[IDs_shuffled[:train_size],:]
        val_label_id_pairs = label_id_pairs[IDs_shuffled[train_size:],:]

        # Validation data
        datasets['val'] = DatasetFromArray(val_label_id_pairs, image_dir, transform=val_transform)
    else:
        train_label_id_pairs = label_id_pairs

    # Training data
    datasets['train'] = DatasetFromArray(train_label_id_pairs, image_dir, transform=transform)

    return datasets


class BatchUnnorm():
    '''Convert normalized batch to original values in [0..1] range.'''
    def __init__(self,mu=[0.485, 0.456, 0.406], sd=[0.229, 0.224, 0.225]):
        '''
        mu: list of means (3 elements, one for each channel)
        sd: list of standard deviations (size same as "mu")
        Default "mu" and "sd" are for ImageNet dataset.
        '''
        self.mu = torch.tensor(mu).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.sd = torch.tensor(sd).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    def __call__(self,img_batch):
        '''Input: batch of images NxCxHxW'''
        return img_batch*self.sd+self.mu


class Unnorm():
    '''Convert normalized (pytorch) image to original values in [0..1] range.'''
    def __init__(self,mu=[0.485, 0.456, 0.406], sd=[0.229, 0.224, 0.225]):
        '''
        mu: list of means (3 elements, one for each channel)
        sd: list of standard deviations (size same as "mu")
        Default "mu" and "sd" are for ImageNet dataset.
        '''
        self.mu = torch.tensor(mu).unsqueeze(-1).unsqueeze(-1)
        self.sd = torch.tensor(sd).unsqueeze(-1).unsqueeze(-1)
    def __call__(self,img_batch):
        '''Input: batch of images CxHxW'''
        return img_batch*self.sd+self.mu

class LungDataset(Dataset):
    '''CS5242 Final Project Dataset (Images look like lung X-ray images). Does not
    have train/val splitting option. {See: DatasetFromArray, and TrainingData}'''
    def __init__(self, csv_file, image_dir, transform=None):
        '''
        Arg-s:
        - csv_file (string): full path to "*.csv" file that contains [ID, label] pairs.
        - image_dir (string): full path to the directory with input images named with "ID.png" pattern.
        - transform (callable): optional transformations on input images {default: None}.
        '''
        super(LungDataset).__init__()
        self.label_id_pairs = pd.read_csv(csv_file).values
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        '''Returns dataset size.'''
        return self.label_id_pairs.shape[0]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_fname = path.join(self.image_dir, f'{self.label_id_pairs[idx,0]}.png')
        x = imread(img_fname)
        y = self.label_id_pairs[idx,1]

        if self.transform:
            x = self.transform(x)

        return {'image':x, 'label': y}
