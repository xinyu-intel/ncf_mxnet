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
import heapq
import random
import numpy as np
import logging
import mxnet as mx

def get_movielens_iter(filename, batch_size, logger):
    """Not particularly fast code to parse the text file and load into NDArrays.
    return two data iters, one for train, the other for validation.
    """
    logger.info("Preparing data iterators for " + filename + " ... ")
    user = []
    item = []
    score = []
    with open(filename, 'r') as f:
        num_samples = 0
        for line in f:
            tks = line.strip().split('\t')
            if len(tks) != 3:
                continue
            if (int(tks[0]) > 138000 or int(tks[1]) > 26700):
                continue
            num_samples += 1
            user.append((tks[0]))
            item.append((tks[1]))
            score.append((tks[2]))
    # convert to ndarrays
    user = mx.nd.array(user, dtype='int32')
    item = mx.nd.array(item)
    score = mx.nd.array(score)
    # prepare data iters
    data = {'user': user, 'item': item}
    label = {'softmax_label': score}
    iter = mx.io.NDArrayIter(data=data, label=label, batch_size=batch_size)
    return iter


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users, num_items = train.shape
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u,j) in train.keys():        
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

def get_train_iters(train, num_negatives, batch_size):
    user, item, label = get_train_instances(train, num_negatives)

    user = mx.nd.array(user, dtype='int32')
    item = mx.nd.array(item, dtype='int32')
    label = mx.nd.array(label) 
    
    data_train = {'user': user, 'item': item}
    label_train = {'softmax_label': label}
    iter_train = mx.io.NDArrayIter(data=data_train,label=label_train,
                                   batch_size=batch_size, shuffle=True)
    return mx.io.PrefetchingIter(iter_train)

def get_eval_iters(testRatings, testNegatives, num_valid, batch_size):
    testUsers=[] 
    testItems=[]
    trueRating=[] #true rating, or posivite item 
    num_items=len(testNegatives[0])+1 #1000
    index=random.sample(range(len(testNegatives)),num_valid) #sample num_valid examples from #test
    for i in range(len(index)):
        _user=[testRatings[i][0]]*num_items
        testNegatives[i].append(testRatings[i][1])
        _item= testNegatives[i]
        _label=testRatings[i][1]
        testUsers.append(_user)
        testItems.append(_item)
        trueRating.append(_label)
    user = mx.nd.array(testUsers, dtype='int32')
    item = mx.nd.array(testItems, dtype='int32')
    data = {'user': user, 'item': item}
    label = mx.nd.array(trueRating, dtype='int32')
    print(batch_size)
    eval_iter = mx.io.NDArrayIter(data=data, label=label, batch_size=batch_size)
    return mx.io.PrefetchingIter(eval_iter), num_items

def evaluate_model(model, eval_iter, num_items, num_valid, K, batch_size):
    print("start evaluting...")
    hits, ndcgs, predictions = [], [], []
    for batch in eval_iter:
        user=batch.data[1].reshape(-1)
        item=batch.data[0].reshape(-1)
        label=mx.nd.zeros(user.shape)
        batch_iter=mx.io.NDArrayIter(data={'user': user, 'item': item}, label=label, batch_size=len(user))

        output = model.predict(batch_iter)
        mx.nd.waitall()
        predictions.append(output)
    eval_iter.reset()
    for b, batch in enumerate(eval_iter):
        user=batch.data[1].reshape(-1)
        item=batch.data[0].reshape(-1)
        for i in range(batch_size):
            map_item_score=dict(zip(item[i*num_items:(i+1)*num_items],predictions[b][i*num_items:(i+1)*num_items].as_in_context(item.context)))
            ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
            hr = getHitRatio(ranklist, batch.label[0][i])
            ndcg = getNDCG(ranklist, batch.label[0][i])
            hits.append(hr)
            ndcgs.append(ndcg)
        print('evaluating batch {} / {}, the length of hits is {} '.format(b, math.ceil(num_valid/batch_size), len(hits)))
        
    return (hits, ndcgs)

def getHitRatio(ranklist, gtItem):
    if gtItem in ranklist:
        return 1
    return 0

def getNDCG(ranklist, gtItem):
    if gtItem in ranklist:
        return math.log(2) / math.log(ranklist.index(gtItem)+2)
    return 0
