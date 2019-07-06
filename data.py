import os
import urllib
import zipfile
import pandas as pd
import numpy as np
import mxnet as mx

def get_dataset(data_dir, num_train = 19000000, batch_size = 25000):
    if not os.path.exists(data_dir + 'ml-20m.zip'):
        os.mkdir(data_dir)
        urllib.request.urlretrieve('http://files.grouplens.org/datasets/movielens/ml-20m.zip', data_dir + 'ml-20m.zip')
    with zipfile.ZipFile(data_dir + "ml-20m.zip", "r") as f:
        f.extractall(data_dir + "./")
    
    data = pd.read_csv(data_dir + './ml-20m/ratings.csv', sep=',', usecols=(0, 1, 2))

    n = num_train
    batch_size = batch_size

    data = data.sample(frac=1).reset_index(drop=True) # Shuffle the data in place row-wise

    max_users = np.unique(data['userId']).shape[0]
    max_items = np.unique(data['movieId']).shape[0]

    train_users = data['userId'].values[:n] - 1 # Offset by 1
    train_movies = data['movieId'].values[:n] - 1 # Offset by 1 
    train_ratings = data['rating'].values[:n]

    valid_users = data['userId'].values[n:] - 1
    valid_movies = data['movieId'].values[n:] - 1
    valid_ratings = data['rating'].values[n:]

    X_train = mx.io.NDArrayIter({'user': train_users, 'item': train_movies}, 
                                label=train_ratings, batch_size=batch_size)
    X_eval = mx.io.NDArrayIter({'user': valid_users, 'item': valid_movies}, 
                               label=valid_ratings, batch_size=batch_size)
    
    return X_train, X_eval, max_users, max_items

