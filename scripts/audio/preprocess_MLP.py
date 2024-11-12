import librosa
import torch
import numpy as np
import pandas as pd
from scipy import stats

from fma.features import columns

def preprocess_MLP(path):

    features = pd.Series(index=columns(), dtype=np.float32)

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    x, sr = librosa.load(path, sr=None, mono=True)

    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7*12, tuning=None))
    f1 = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cqt', f1)

    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    f2 = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    feature_stats('mfcc', f2)

    data_mfcc = features.loc['mfcc']
    data_cqt = features.loc['chroma_cqt']

    data_mfcc = torch.tensor(data_mfcc.values, dtype = torch.float32)
    data_cqt =  torch.tensor(data_cqt.values, dtype = torch.float32)
    song_data = torch.cat((data_mfcc, data_cqt), dim = -1)

    return song_data