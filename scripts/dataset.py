from torch.utils.data import Dataset
import torch
import fma.utils
import numpy as np
import pandas as pd
from genre_utils import new_id

# Global label mapping dictionary for known labels in training
global_label_mapping = {}
unknown_label = -1  # Label for unknown genres in validation and test sets

class FMA_Dataset(Dataset):
    '''
    Class representing the FMA dataset for using in a DataLoader
    '''

    def __init__(self, split, subset, mode="top"):
        '''
        Initializes the dataset object.

        Args:
          split: string, either training, validation or test, part of the dataset to be
                 returned
          subset: string, either small, medium or large, size of the dataset
          mode: string, either top or all, determines if only the top genre or all songs 
                genres are used for classification training
        '''
        
        self.split = split
        self.subset = subset
        self.mode = mode
        self.data, self.labels = self.preprocess()
        
        self.n_inputs = self.data.shape[1]
        self.n_classes = len(torch.unique(self.labels))
        self.genre_names = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def preprocess(self):
        '''
        Preprocesses the data to be stored as torch tensors
        '''
        
        if self.split not in ["training", "validation", "test"]:
            raise ValueError("Provide a correct split: 'training', 'validation', or 'test'")
        
        if self.subset not in ["small", "medium", "large"]:
            raise ValueError("Provide a correct subset: 'small', 'medium', or 'large'")
        
        if self.mode not in ["top", "all"]:
            raise ValueError("Provide a correct mode: 'top' or 'all'")
        
        tracks = fma.utils.load('data/fma_metadata/tracks.csv')
        features = fma.utils.load('data/fma_metadata/features.csv')

        data_split = tracks['set', 'split'] == self.split
        data_subset = tracks['set', 'subset'] <= self.subset
        
        data_mfcc = features.loc[data_split & data_subset, 'mfcc']
        data_cqt = features.loc[data_split & data_subset, 'chroma_cqt']
        

        if self.mode == "top":
            labels = tracks.loc[data_split & data_subset, ('track', 'genre_top')]
            print(labels[labels.isna()].index)
            labels = labels.dropna()
            self.genre_names = np.unique(labels.values)
            labels = labels.apply(lambda x: new_id(x) if x and isinstance(x, str) and len(x) > 0 else unknown_label)
            data_mfcc = data_mfcc[data_mfcc.index.isin(labels.index)]
            data_cqt = data_cqt[data_cqt.index.isin(labels.index)]
        else: 
            # TODO: handle multiple class labels
            pass

        labels = torch.tensor(labels.values, dtype = int)
        data_mfcc = torch.tensor(data_mfcc.values, dtype = torch.float32)
        data_cqt =  torch.tensor(data_cqt.values, dtype = torch.float32)
        data = torch.cat((data_mfcc, data_cqt), dim = 1)

        return data, labels
