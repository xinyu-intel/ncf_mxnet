import numpy as np
import mxnet as mx
import heapq
import math
import random
from Dataset import Dataset

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

def get_eval_iters(user, item, label, batch_size):
    user = mx.nd.array(user, dtype='int32')
    item = mx.nd.array(item, dtype='int32')
    data = {'user': user, 'item': item}
    label = mx.nd.array(label, dtype='int32')
    eval_iter = mx.io.NDArrayIter(data=data, label=label, batch_size=batch_size)
    return mx.io.PrefetchingIter(eval_iter)


def evaluate_model(model, testRatings, testNegatives, K, num_valid, batch_size):
    testUsers=[] 
    testItems=[]
    trueRating=[] #true rating, or posivite item 
    num_items=len(testNegatives[0]) #1000
    index=random.sample(range(len(testNegatives)),num_valid) #sample num_valid examples from #test
    for i in range(len(index)):
        _user=[testRatings[i][0]]*num_items
        _item=testNegatives[i]
        _label=testRatings[i][1]
        testUsers.append(_user)
        testItems.append(_item)
        trueRating.append(_label)
    eval_iter=get_eval_iters(testUsers, testItems, trueRating, batch_size)
    print("start evaluting...")
    hits, ndcgs = [],[]
    nbatch=0
    for batch in eval_iter:
        user=batch.data[1].reshape(-1)
        item=batch.data[0].reshape(-1)
        label=mx.nd.zeros(user.shape)
        # batch_iter=mx.io.NDArrayIter(data={'user': user, 'item': item}, label=label, batch_size=batch_size*num_items)
        batch_iter=mx.io.NDArrayIter(data={'user': user, 'item': item}, label=label, batch_size=len(user))

        predictions = model.predict(batch_iter) 
        for i in range(batch_size):
            map_item_score=dict(zip(item[i*num_items:(i+1)*num_items],predictions[i*num_items:(i+1)*num_items].as_in_context(item.context)))
            ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
            hr = getHitRatio(ranklist, batch.label[0][i])
            ndcg = getNDCG(ranklist, batch.label[0][i])
            hits.append(hr)
            ndcgs.append(ndcg)
        nbatch+=1
        print('evaluating batch {} / {}, the length of hits is {} '.format(nbatch, math.ceil(num_valid/batch_size), len(hits)))
    
    
    return (hits, ndcgs)


def getHitRatio(ranklist, gtItem):
    if gtItem in ranklist:
        return 1
    return 0
    # for item in ranklist:
    #     if item == gtItem:
    #         return 1
    # return 0

def getNDCG(ranklist, gtItem):
    if gtItem in ranklist:
        return math.log(2) / math.log(ranklist.index(gtItem)+2)
    return 0
    # for i in range(len(ranklist)):
    #     item = ranklist[i]
    #     if item == gtItem:
    #         return math.log(2) / math.log(i+2)
    # return 0

if __name__ == "__main__":
    data = Dataset('mini_data/ml-20m')
    train, testRatings, testNegatives = data.trainMatrix, data.testRatings, data.testNegatives
    train_iter = get_train_iters(train, 4, 98304)
    net, arg_params, aux_params = mx.model.load_checkpoint('model/ml-20m/neumf', 0)
    mod = mx.module.Module(net, data_names=['user', 'item'], label_names=['softmax_label'])
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    mod.set_params(arg_params, aux_params)
    evaluate_model(mod, testRatings, testNegatives, 10, 100, 10)




