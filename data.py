import os
import urllib
import zipfile
import pandas as pd
import numpy as np
import mxnet as mx
import multiprocessing

np.random.seed(1234)

def get_movielens_data(data_dir, dataset):
    if not os.path.exists(data_dir + '%s.zip' % dataset):
        os.mkdir(data_dir)
        urllib.request.urlretrieve('http://files.grouplens.org/datasets/movielens/%s.zip' % dataset, data_dir + dataset + '.zip')
    with zipfile.ZipFile(data_dir + "%s.zip" % dataset, "r") as f:
        f.extractall(data_dir + "./")

data_dir = './data/'
num_train = 19000000
get_movielens_data(data_dir, 'ml-20m')
data = pd.read_csv(data_dir + './ml-20m/ratings.csv', sep=',', usecols=(0, 1, 2))
data = data.sample(frac=1).reset_index(drop=True) # Shuffle the data in place row-wise
max_users = np.unique(data['userId']).shape[0]
max_items = np.unique(data['movieId']).shape[0]

train_data = data[:num_train]
train_data['userId'] = train_data.loc[:,'userId'] - 1 # Offset by 1
train_data['movieId'] = train_data.loc[:,'movieId'] - 1 # Offset by 1 
valid_data = data[num_train:]
valid_data['userId'] = valid_data.loc[:,'userId'] - 1
valid_data['movieId'] = valid_data.loc[:,'movieId'] - 1

train_data.to_csv(data_dir + './ml-20m/ml-20m.train.rating', sep='\t', header=False, index=False)
valid_data.to_csv(data_dir + './ml-20m/ml-20m.test.rating', sep='\t', header=False, index=False)

val_data = valid_data.as_matrix()

def write_file(index=0):
    with open(data_dir + './ml-20m/ml-20m.test.negative2', "a+") as f:
        f.write('('+str(val_data[index,0])+str(',')+str(val_data[index,1])+')')
        for t in range(99):
            j = np.random.randint(max_items)
            while val_data[(val_data[:,0]==index) & (val_data[:,1]==j)].shape[0] != 0:#np.array([val_data[i,0], j]) in val_data[:,:2]:
                j = np.random.randint(max_items)
            f.write('\t'+str(j))
        f.write('\n')

def process(index=0):
    '''
    work
    '''
    return index

def prepare_dataset_ml20m(data_dir, num_train = 19000000):

    get_movielens_data(data_dir, 'ml-20m')
    data = pd.read_csv(data_dir + './ml-20m/ratings.csv', sep=',', usecols=(0, 1, 2))
    data = data.sample(frac=1).reset_index(drop=True) # Shuffle the data in place row-wise
    max_users = np.unique(data['userId']).shape[0]
    max_items = np.unique(data['movieId']).shape[0]

    train_data = data[:num_train]
    train_data['userId'] = train_data.loc[:,'userId'] - 1 # Offset by 1
    train_data['movieId'] = train_data.loc[:,'movieId'] - 1 # Offset by 1 
    valid_data = data[num_train:]
    valid_data['userId'] = valid_data.loc[:,'userId'] - 1
    valid_data['movieId'] = valid_data.loc[:,'movieId'] - 1

    # train_data.to_csv(data_dir + './ml-20m/ml-20m.train.rating2', sep=' ', header=False)
    # valid_data.to_csv(data_dir + './ml-20m/ml-20m.test.rating2', sep=' ', header=False)
    
    val_data = valid_data.as_matrix()

    def write_file(index=0):
        with open(data_dir + './ml-20m/ml-20m.test.negative2', "a+") as f:
            f.write('('+str(val_data[index,0])+str(',')+str(val_data[index,1])+')')
            for t in range(99):
                j = np.random.randint(max_items)
                while val_data[(val_data[:,0]==index) & (val_data[:,1]==j)].shape[0] != 0:#np.array([val_data[i,0], j]) in val_data[:,:2]:
                    j = np.random.randint(max_items)
                f.write('\t'+str(j))
            f.write('\n')

    def process(index=0):
        '''
        work
        '''
        return index
    
    pool = multiprocessing.Pool(processes = 100)
    for i in range(val_data.shape[0]):
        pool.apply_async(process, (i, ), callback=write_file)
    pool.close()
    pool.join()
    # with open(data_dir + './ml-20m/ml-20m.test.negative2', "a+") as f:
    #     for i in range(val_data.shape[0]):
    #         f.write('('+str(val_data[i,0])+str(',')+str(val_data[i,1])+')')
    #         for t in range(99):
    #             j = np.random.randint(max_items)
    #             while val_data[(val_data[:,0]==i) & (val_data[:,1]==j)].shape[0] != 0:#np.array([val_data[i,0], j]) in val_data[:,:2]:
    #                 j = np.random.randint(max_items)
    #             f.write('\t'+str(j))
    #         f.write('\n')

if __name__ == "__main__":
    print("a")
    # prepare_dataset_ml20m('./data/')
    # pool = multiprocessing.Pool(processes = 100)
    # for i in range(val_data.shape[0]):
    #     pool.apply_async(process, (i, ), callback=write_file)
    # pool.close()
    # pool.join()
