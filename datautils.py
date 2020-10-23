import torch
import numpy as np
import pandas as pd
from skimage.io import imread
from os import path
from torch.utils.data import Dataset, DataLoader

class LungDataset(Dataset):
    '''
    CS5242 Final Project Dataset Loader.

    - Docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    - Tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
    '''
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
