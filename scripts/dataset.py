from torch.utils.data import Dataset
import torch
import fma.utils
import numpy as np
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
        self.n_classes = self.convert_labels()

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
        
        data = features.loc[data_split & data_subset, 'mfcc']
        

        if self.mode == "top":
            labels = tracks.loc[data_split & data_subset, ('track', 'genre_top')]
            labels = labels.apply(lambda x: new_id(x) if x and isinstance(x, str) and len(x) > 0 else unknown_label)
            labels = labels.dropna()
            data = data[data.index.isin(labels.index)]
        else: 
            # TODO: handle multiple class labels
            pass

        labels = torch.tensor(labels.values, dtype = int)
        data = torch.tensor(data.values, dtype = torch.float32)

        return data, labels
    
    def convert_labels(self):
        global global_label_mapping, unknown_label

        if self.split == "training" and not global_label_mapping:

            unique_labels = torch.unique(self.labels)
            global_label_mapping = {old_label.item(): idx for idx, old_label in enumerate(unique_labels)}
        
        # Map labels based on training's label mapping; unknown labels get the `unknown_label` value
        self.labels = torch.tensor(
            [global_label_mapping.get(label.item(), unknown_label) for label in self.labels], 
            dtype = int
        )
        
        return len(global_label_mapping)
