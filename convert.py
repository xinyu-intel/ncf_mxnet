import os
import logging
import urllib
import zipfile
import argparse
import pandas as pd
import numpy as np
import mxnet as mx
import multiprocessing

np.random.seed(1234)

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="convert movielens data for ncf model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', nargs='?', default='./data/',
                    help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='ml-20m', choices=['ml-20m'],
                    help='The dataset name, temporary support ml-20m.')
parser.add_argument('--num-train', type=int, default=19000000,
                    help='number of data for training')
parser.add_argument('--no-negative', action='store_true', help="write no negative examples")
parser.add_argument('--negative-num', type=int, default=99,
                    help='number of negatives per example')

def get_movielens_data(data_dir, dataset):
    if not os.path.exists(data_dir + '%s.zip' % dataset):
        os.mkdir(data_dir)
        urllib.request.urlretrieve('http://files.grouplens.org/datasets/movielens/%s.zip' % dataset, data_dir + dataset + '.zip')
    with zipfile.ZipFile(data_dir + "%s.zip" % dataset, "r") as f:
        f.extractall(data_dir + "./")

def write_negative_examples(filename, val_data, max_items, negative_num=99):
    val_data = valid_data.as_matrix()
    with open(filename, "a+") as f:
        for index in range(val_data.shape[0]):
            f.write('('+str(val_data[index,0])+str(',')+str(val_data[index,1])+')')
            for t in range(99):
                j = np.random.randint(max_items)
                while val_data[(val_data[:,0]==index) & (val_data[:,1]==j)].shape[0] != 0:
                    j = np.random.randint(max_items)
                f.write('\t'+str(j))
            f.write('\n')

if __name__ == '__main__':

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)

    data_dir = args.path
    num_train = args.num_train
    dataset = args.dataset
    negative_num = args.negative_num
    logging.info('download movielens %s dataset' % dataset)
    get_movielens_data(data_dir, dataset)
    data = pd.read_csv(data_dir + dataset + '/ratings.csv', sep=',', usecols=(0, 1, 2))
    data = data.sample(frac=1).reset_index(drop=True) # Shuffle the data in place row-wise
    max_users = np.unique(data['userId']).shape[0]
    max_items = np.unique(data['movieId']).shape[0]

    train_data = data[:num_train]
    train_data['userId'] = train_data.loc[:,'userId'] - 1 # Offset by 1
    train_data['movieId'] = train_data.loc[:,'movieId'] - 1 # Offset by 1 
    valid_data = data[num_train:]
    valid_data['userId'] = valid_data.loc[:,'userId'] - 1
    valid_data['movieId'] = valid_data.loc[:,'movieId'] - 1

    logging.info('save training dataset into %s' % (data_dir + dataset + '/ml-20m.train.rating'))
    train_data.to_csv(data_dir + dataset + '/ml-20m.train.rating', sep='\t', header=False, index=False)
    logging.info('save validation dataset into %s' % (data_dir + dataset + '/ml-20m.test.rating'))
    valid_data.to_csv(data_dir + dataset + '/ml-20m.test.rating', sep='\t', header=False, index=False)

    if not args.no_negative:
        logging.info('save negative dataset into %s' % (data_dir + dataset + '/ml-20m.test.negative'))
        write_negative_examples(data_dir + dataset + '/ml-20m.test.negative',
                                val_data=valid_data, max_items=max_items, negative_num=negative_num)
