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
import os
import time
import argparse
import logging
import numpy as np
import mxnet as mx
from model import get_model
from Dataset import Dataset
from evaluate import evaluate_model

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Run matrix factorization with embedding",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', nargs='?', default='./data/',
                    help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='ml-20m',
                    help='The dataset name.')
parser.add_argument('--num-epoch', type=int, default=3,
                    help='number of epochs to train')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--batch-size', type=int, default=256,
                    help='number of examples per batch')
parser.add_argument('--log-interval', type=int, default=100,
                    help='logging interval')
parser.add_argument('--num-neg', type=int, default=4,
                    help='Number of negative instances to pair with a positive instance.')
parser.add_argument('--num-valid', type=int, default=1000,
                    help='Number of validation examples to validate per training epoch.')
parser.add_argument('--model-type', type=str, default='neumf', choices=['neumf', 'gmf', 'mlp'],
                    help="mdoel type")
parser.add_argument('--layers', type=list, default=[256, 128, 64],
                    help="list of number hiddens of fc layers in mlp model.")
parser.add_argument('--factor-size-mlp', type=int, default=128,
                    help="outdim of mlp embedding layers.")
parser.add_argument('--factor-size-gmf', type=int, default=64,
                    help="outdim of gmf embedding layers.")
parser.add_argument('--num-hidden', type=int, default=1,
                    help="num-hidden of neumf fc layer")
parser.add_argument('--gpus', type=str,
                    help="list of gpus to run, e.g. 0 or 0,2. empty means using cpu().")
parser.add_argument('--sparse', action='store_true', help="whether to use sparse embedding")


def get_train_instances(train, num_negatives): # train是scipy.sparse.dok_matrix类型
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


if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)
    
    num_epoch = args.num_epoch
    num_negatives = args.num_neg
    num_valid = args.num_valid
    batch_size = args.batch_size
    model_type = args.model_type
    factor_size_mlp = args.factor_size_mlp
    factor_size_gmf = args.factor_size_gmf
    model_layers = args.layers
    num_hidden = args.num_hidden
    sparse = args.sparse
    log_interval = args.log_interval

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus else mx.cpu()
    optimizer = 'sgd'
    learning_rate = 0.1
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    topK = 10
    evaluation_threads = 1

    # prepare dataset and iterators
    data = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = data.trainMatrix, data.testRatings, data.testNegatives
    max_user, max_movies = train.shape
    print("Load data done. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(max_user, max_movies, train.nnz, len(testRatings)))
    train_iter = get_train_iters(train, num_negatives, batch_size)

    # construct the model
    net = get_model(model_type, factor_size_mlp, factor_size_gmf, 
                    model_layers, num_hidden, max_user, max_movies, sparse)

    # initialize the module
    mod = mx.module.Module(net, context=ctx, data_names=['user', 'item'], label_names=['softmax_label'])
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    mod.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    optim = mx.optimizer.Adam()
    mod.init_optimizer(optimizer=optim, kvstore='device')
    # use binary cross entropy as the metric
    def cross_entropy(label, pred):
        ce = 0
        for l, p in zip(label, pred):
            ce += -( l*np.log(p) + (1-l)*np.log(1-p))
        return ce
    
    metric = mx.metric.create(cross_entropy)

    speedometer = mx.callback.Speedometer(batch_size, log_interval)
    
    best_hr, best_ndcg, best_iter = -1, -1, -1 
    logging.info('Training started ...')
    for epoch in range(num_epoch):
        nbatch = 0
        metric.reset()
        for batch in train_iter:
            nbatch += 1
            mod.forward(batch)
            pred = mod.get_outputs()[0]
            mod.backward()
            mod.update()
            label = batch.label[0]
            metric.update(label, pred)
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=metric, locals=locals())
            speedometer(speedometer_param)
        # reset iterator
        train_iter.reset()
        # save model
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, 'model', args.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        mod.save_checkpoint(os.path.join(model_path, model_type), epoch)
        # compute hit ratio
        (hits, ndcgs) = evaluate_model(mod, testRatings[0:num_valid], testNegatives[0:num_valid], topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        
        logging.info('Iteration %d: HR = %.4f, NDCG = %.4f'  % (epoch, hr, ndcg))
        # best hit ratio
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch

    logging.info("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    logging.info('Training completed.')
