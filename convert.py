# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# 
import math
import os
import logging
import urllib
import zipfile
import argparse
import pandas as pd
import numpy as np
import mxnet as mx
import multiprocessing

from argparse import ArgumentParser
from load import implicit_load

# from logger.logger import LOGGER
# from logger import tags


MIN_RATINGS = 20
USER_COLUMN = 'user_id'
ITEM_COLUMN = 'item_id'

# LOGGER.model = 'ncf'

# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument('--path', type=str, default='./data/ml-20m/ratings.csv',
#                         help='Path to reviews CSV file from MovieLens')
#     parser.add_argument('--output', type=str, default='./data',
#                         help='Output directory for train and test files')
#     return parser.parse_args()

np.random.seed(1234)

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="convert movielens data for ncf model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', nargs='?', default='./data/',
                    help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='ml-20m', choices=['ml-20m'],
                    help='The dataset name, temporary support ml-20m.')
parser.add_argument('--no-negative', action='store_true', help="write no negative examples")
parser.add_argument('--negative-num', type=int, default=999,
                    help='number of negatives per example')
                    



def get_movielens_data(data_dir, dataset):
    if not os.path.exists(data_dir + '%s.zip' % dataset):
        os.mkdir(data_dir)
        urllib.request.urlretrieve('http://files.grouplens.org/datasets/movielens/%s.zip' % dataset, data_dir + dataset + '.zip')
        with zipfile.ZipFile(data_dir + "%s.zip" % dataset, "r") as f:
            f.extractall(data_dir + "./")

def write_negative_examples(filename, val_data, max_items, negative_num=999):
    f=open(filename,'a')
    epoch=math.ceil(float(val_data.shape[0])/1000) # number of blocks
    for i in range(epoch):
        print("\r epoch %d" % i)
        start=i*1000
        if i == epoch-1:
            end=-1 # in last cycle, the block is less then 1000
        end=(i+1)*1000
        val_mat = val_data[start:end].as_matrix().astype(np.int)[:,:2]
        neg_mat=[]
        for index in range(val_mat.shape[0]):
            neg_data=[]
            pos_items=np.repeat(val_mat[index,1], negative_num) # 999个positve item
            while len(pos_items) > 0: 
                neg_items = np.random.randint(0, high=max_items, size=len(pos_items))
                neg_mask = pos_items != neg_items # logical == 
                neg_data.append(neg_items[neg_mask])
                
                pos_items = pos_items[np.logical_not(neg_mask)]
            neg_data=np.concatenate(neg_data)
            neg_mat.append(neg_data)

        neg_mat = np.hstack([val_mat,neg_mat]) # concatanate
        np.savetxt(f, neg_mat, delimiter='\t', fmt="%d")
    f.close()


if __name__ == '__main__':

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)

    data_dir = args.path
    dataset = args.dataset
    negative_num = args.negative_num
    logging.info('download movielens %s dataset' % dataset)
    get_movielens_data(data_dir, dataset)
    # data = pd.read_csv(data_dir + dataset + '/ratings.csv', sep=',', usecols=(0, 1, 2)) # DataFrame 
    # data = data.sample(frac=1).reset_index(drop=True) # Shuffle the data in place row-wise
    # max_users = np.unique(data['userId']).shape[0]
    # max_items = np.unique(data['movieId']).shape[0]

    # train_data = data[:num_train]
    # train_data['userId'] = train_data.loc[:,'userId'] - 1 # Offset by 1
    # train_data['movieId'] = train_data.loc[:,'movieId'] - 1 # Offset by 1 
    # valid_data = data[num_train:]
    # valid_data['userId'] = valid_data.loc[:,'userId'] - 1
    # valid_data['movieId'] = valid_data.loc[:,'movieId'] - 1


    print("Loading raw data from {}/ratings.csv".format(data_dir + dataset))
    df = implicit_load(data_dir + dataset + '/ratings.csv', sort=False)

    print("Filtering out users with less than {} ratings".format(MIN_RATINGS))
    grouped = df.groupby(USER_COLUMN)
    # LOGGER.log(key=tags.PREPROC_HP_MIN_RATINGS, value=MIN_RATINGS)
    df = grouped.filter(lambda x: len(x) >= MIN_RATINGS)

    print("Mapping original user and item IDs to new sequential IDs")
    df[USER_COLUMN] = pd.factorize(df[USER_COLUMN])[0]
    df[ITEM_COLUMN] = pd.factorize(df[ITEM_COLUMN])[0]

    print("Creating list of items for each user")
    # Need to sort before popping to get last item
    df.sort_values(by='timestamp', inplace=True)

    # clean up data
    del df['rating'], df['timestamp']
    df = df.drop_duplicates() # assuming it keeps order

    # now we have filtered and sorted by time data, we can split test data out
    grouped_sorted = df.groupby(USER_COLUMN, group_keys=False)
    test_data = grouped_sorted.tail(1).sort_values(by='user_id')
    # need to pop for each group
    train_data = grouped_sorted.apply(lambda x: x.iloc[:-1])
    train_data = train_data.sort_values([USER_COLUMN, ITEM_COLUMN])

    max_users, max_items = train_data.max() + 1

    # train_data.to_pickle(args.output + '/train_ratings.pickle')
    # test_data.to_pickle(args.output + '/test_ratings.pickle')

    logging.info('save training dataset into %s' % (data_dir + dataset + '.train.rating'))
    train_data.to_csv(data_dir + dataset + '.train.rating', sep='\t', header=False, index=False)
    logging.info('save validation dataset into %s' % (data_dir + dataset + '.test.rating'))
    test_data.to_csv(data_dir + dataset + '.test.rating', sep='\t', header=False, index=False)

    if not args.no_negative:
        logging.info('save negative dataset into %s' % (data_dir + dataset + '.test.negative'))
        write_negative_examples(data_dir + dataset + '.test.negative',
                                val_data=test_data, max_items=max_items, negative_num=negative_num)
