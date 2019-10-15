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

def get_movielens_iter(filename, batch_size, ctx, logger):
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
    user = mx.nd.array(user, dtype='int32').as_in_context(ctx)
    item = mx.nd.array(item, dtype='int32').as_in_context(ctx)
    score = mx.nd.array(score).as_in_context(ctx)
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

def get_train_iters(train, num_negatives, batch_size, ctx):
    user, item, label = get_train_instances(train, num_negatives)

    user = mx.nd.array(user, dtype='int32').as_in_context(ctx)
    item = mx.nd.array(item, dtype='int32').as_in_context(ctx)
    label = mx.nd.array(label).as_in_context(ctx)
    
    data_train = {'user': user, 'item': item}
    label_train = {'softmax_label': label}
    iter_train = mx.io.NDArrayIter(data=data_train,label=label_train,
                                   batch_size=batch_size, shuffle=True)
    return mx.io.PrefetchingIter(iter_train)

def predict(model, users, items, batch_size=1000, ctx=mx.cpu()):
    user = mx.nd.array(users, dtype='int32').as_in_context(ctx)
    item = mx.nd.array(items, dtype='int32').as_in_context(ctx)
    label = mx.nd.zeros(len(user)).as_in_context(ctx)
    data = {'user': user, 'item': item}
    label = {'softmax_label':label}
    eval_iter = mx.io.NDArrayIter(data=data, label=label, batch_size=batch_size)
    preds = []
    for batch in eval_iter:
        model.forward(batch)
        mx.nd.waitall()
        outp = model.get_outputs()[0].asnumpy()
        preds += list(outp.flatten())
    return preds

def _calculate_hit(ranked, test_item):
    return int(test_item in ranked)


def _calculate_ndcg(ranked, test_item):
    for i, item in enumerate(ranked):
        if item == test_item:
            return math.log(2) / math.log(i + 2)
    return 0.

def eval_one(rating, items, model, K, ctx):
    user = rating[0]
    test_item = rating[1]
    items.append(test_item)
    users = [user] * len(items)
    predictions = predict(model, users, items, ctx)

    map_item_score = {item: pred for item, pred in zip(items, predictions)}
    ranked = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    hit = _calculate_hit(ranked, test_item)
    ndcg = _calculate_ndcg(ranked, test_item)
    return hit, ndcg, len(predictions)

def evaluate_model(model, ratings, negs, K, batch_size, ctx):
    hits, ndcgs, num_preds = [], [], []
    index = 0
    for rating, items in zip(ratings, negs):
        index += 1
        hit, ndcg, num_pred = eval_one(rating, items, model, K, ctx)
        hits.append(hit)
        ndcgs.append(ndcg)
        num_preds.append(num_pred)
        if index % batch_size == 0:
            print('evaluating test data {} / {}'.format(index, len(ratings)))

    return hits, ndcgs
