import fma.utils
import numpy as np

'''
This script contrains utility functions for dealing with genre names.
'''

genre_names = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic',
    'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Jazz',
    'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']

genre_dict = {genre: idx for idx, genre in enumerate(genre_names)}

def new_id(name):

    new_id = genre_dict[name]
    return new_id
