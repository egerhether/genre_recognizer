import fma.utils
import numpy as np

genre_names = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic',
    'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Jazz',
    'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']

genre_dict = {genre: idx for idx, genre in enumerate(genre_names)}

def genres():

    genres = fma.utils.load('data/fma_metadata/genres.csv')
    genres = genres['title']
    genres = genres.set_axis(np.arange(0, len(genres.unique())))

    return genres

def genre_to_id(genre):

    genres_df = genres()
    id = genres[genres_df == genre].index[0]

    return id

def id_to_genre(id):

    genres_df = genres()
    genre = genres_df[id]

    return genre

def new_id(name):

    new_id = genre_dict[name]
    return new_id

def old_id(new_id):

    genres_df = fma.utils.load('data/fma_metadata/genres.csv')
    genres_df = genres_df['title']
    name = genres()[new_id]
    old_id = genres_df[genres_df == name].index[0]

    return old_id