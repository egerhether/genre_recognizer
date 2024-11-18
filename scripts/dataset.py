from torch.utils.data import Dataset
import torch
import fma.utils
import numpy as np
import pandas as pd
import glob
import librosa
import os
import warnings
from tqdm import tqdm
from genre_utils import new_id

# Run this script after downloading the fma_{size}.zip dataset and extracting it do
# data/fma_{size}


warnings.filterwarnings("error", category=UserWarning)

class FMA_Dataset(Dataset):
    '''
    Class representing the FMA dataset for using in a DataLoader
    '''

    def __init__(self, split, subset, arch, mode="top"):
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
        self.arch = arch
        self.data, self.labels = self.preprocess()
        
        self.n_inputs = self.data.shape[1] if arch == "mlp" else self.data.shape[2]
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
        data_cens = features.loc[data_split & data_subset, 'chroma_cens']
        data_stft = features.loc[data_split & data_subset, 'chroma_stft']
        

        if self.mode == "top":
            labels = tracks.loc[data_split & data_subset, ('track', 'genre_top')]
            labels = labels.dropna()
            self.genre_names = np.unique(labels.values)
            labels = labels.apply(lambda x: new_id(x))
            data_mfcc = data_mfcc[data_mfcc.index.isin(labels.index)]
            data_cqt = data_cqt[data_cqt.index.isin(labels.index)]
            data_cens = data_cens[data_cens.index.isin(labels.index)]
            data_stft = data_stft[data_stft.index.isin(labels.index)]
        else: 
            # TODO: handle multiple class labels
            pass

        labels = torch.tensor(labels.values, dtype = int)
        data_mfcc = torch.tensor(data_mfcc.values, dtype = torch.float32)
        data_cqt = torch.tensor(data_cqt.values, dtype = torch.float32)
        data_cens = torch.tensor(data_cens.values, dtype = torch.float32)
        data_stft = torch.tensor(data_stft.values, dtype = torch.float32)
        data = torch.cat((data_mfcc, data_cqt, data_cens, data_stft), dim = 1)

        if self.arch == "cnn":
            data = torch.reshape(data, (data.shape[0], 1, data.shape[1]))

        return data, labels
    


class FMA_Audio_Dataset(Dataset):
    '''
    Class representing the audio vectors of the FMA Dataset for use in a dataloader
    '''

    def __init__(self, split, source = "data/audio_data/*.npz"):

        self.files = glob.glob(source)
        test_size = len(self.files) // 7
        train_size = len(self.files) - 2 * test_size

        if split == "training":
            self.files = self.files[:train_size]
        elif split == "test":
            self.files = self.files[train_size : (train_size + test_size)]
        elif split == "validation":
            self.files = self.files[(train_size + test_size) : (train_size + 2 * test_size)]

        self.current_batch = 0
        self.data, self.labels = self.load_batches()
        self.total_length = 0

    def __getitem__(self, index):
            
        return self.data[index], self.labels[index]
    
    def load_batches(self):
        '''
        Load the next .npz file
        '''

        data = torch.tensor([])
        labels = torch.tensor([])

        for file in self.files:
            loaded = np.load(file)
            data = torch.cat((data, torch.tensor(loaded['x'])), 0)
            labels = torch.cat((labels, torch.tensor(loaded['y'])), 0)

        data = torch.reshape(data, (data.shape[0], 1, data.shape[1]))

        return data, labels.type(torch.LongTensor)
    
    def __len__(self):
        return len(self.data)


def create_dataset(batch_size = 500):
    '''
    Function loading the fma audio files and saving their decoded vectorized representation in a torch.Tensor.
    Saves batches of the dataset in order to allow for lower memory systems to generate the dataset.
    '''

    audio_files = glob.glob("data/fma_medium/*/*.mp3")
    tracks = fma.utils.load('data/fma_metadata/tracks.csv')
    audio = []
    labels = []
    batch_counter = 0

    for idx, file in enumerate(tqdm(audio_files)):
        track_id = int(os.path.basename(file)[:-4])

        try:
            x, sr = librosa.load(file, sr = 10100 / 30)
            if x.shape[0] != 10000:
                x = x[:10000]
                if x.shape[0] != 10000:
                    raise UserWarning
                
            stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
            mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
            x = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
            audio.append(x)
            label = new_id(tracks.loc[track_id, ('track', 'genre_top')])
            labels.append(label)

        except UserWarning:
            # this is necessary as sometimes a file might be corrupted
            print(f"Error loading {file}")
            continue

        if len(audio) % batch_size == 0 or (idx + 1) == len(audio_files):
            audio_data = np.array(audio)
            labels = np.array(labels)
            np.savez_compressed(f"data/audio_data/audio_data_{batch_counter}", x = audio_data, y = labels)

            audio = []
            labels = []
            batch_counter += 1


if __name__ == "__main__":

    create_dataset(2500)