import torch
import os
import torch.nn.functional as F
import warnings

from audio.preprocess_MLP import preprocess_MLP
from MLP import MLP

def main():

    genre_names = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic',
    'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Jazz',
    'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']

    # TODO: read this from file
    model = MLP(224, [164], 16, True, 0.3)

    pretrained_state_dict = torch.load("trained_models/MLP_5958.pth", map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(parent_dir, "input/input_song.mp3")

    model.to(device)

    input_song = preprocess_MLP(path)
    input_song = input_song.to(device)
    input_song = input_song.unsqueeze(0)

    model.eval()

    with torch.no_grad():
        
        genre_pred = model.forward(input_song)
        genre_prob = F.softmax(genre_pred)[0].tolist()
        
        print("-------------------------------------------------------------")
        print("   Predictions (displaying only if confidence is above 20%)")
        print("-------------------------------------------------------------")

        for genre, p in zip(genre_names, genre_prob):
            if p > 0.2:
                print(f"{genre}: {p * 100:2f}%")



if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=FutureWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    main()