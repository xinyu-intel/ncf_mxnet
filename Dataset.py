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
import scipy.sparse as sp
import numpy as np
import pandas as pd

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path, is_train = False):
        '''
        Constructor
        '''
        self.num_users = 138493
        self.num_items = 26744
        self.testRatings = self.load_rating_file_as_list(path + "/test-ratings.csv")
        self.testNegatives = self.load_negative_file(path + "/test-negative.csv")
        assert len(self.testRatings) == len(self.testNegatives)
        if is_train:
            self.data_raw = pd.read_csv(path + "/ratings.csv", sep=',', usecols=(0, 1, 2))
            self.trainMatrix = self.load_rating_file_as_matrix(path + "/train-ratings.csv")
            self.num_users = np.unique(self.data_raw['userId']).shape[0] #138493
            self.num_items = np.unique(self.data_raw['movieId']).shape[0] #26744


    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Construct matrix
        mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                mat[user, item] = 1.0
                line = f.readline()    
        return mat
