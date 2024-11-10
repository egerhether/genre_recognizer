import fma.utils
import numpy as np

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

    genres_df = fma.utils.load('data/fma_metadata/genres.csv')
    genres_df = genres_df['title']
    temp = genres()
    new_id = temp[temp == name].index[0]

    return new_id

def old_id(new_id):

    genres_df = fma.utils.load('data/fma_metadata/genres.csv')
    genres_df = genres_df['title']
    name = genres()[new_id]
    old_id = genres_df[genres_df == name].index[0]

    return old_id