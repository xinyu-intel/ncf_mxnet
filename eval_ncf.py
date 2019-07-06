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
import mxnet as mx
from data import get_dataset
from model import get_model

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Run matrix factorization with sparse embedding",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', nargs='?', default='./data/',
                    help='Input data path.')                                
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--batch-size', type=int, default=256,
                    help='number of examples per batch')
parser.add_argument('--log-interval', type=int, default=100,
                    help='logging interval')
parser.add_argument('--num_neg', type=int, default=4,
                    help='Number of negative instances to pair with a positive instance.')
parser.add_argument('--model-type', type=str, default='neumf', choices=['neumf', 'gmf', 'mlp'],
                    help="mdoel type")
parser.add_argument('--layers', type=list, default=[256, 128, 64],
                    help="list of number hiddens of fc layers in mlp model.")
parser.add_argument('--factor_size_mlp', type=int, default=128,
                    help="outdim of mlp embedding layers.")
parser.add_argument('--factor_size_gmf', type=int, default=64,
                    help="outdim of gmf embedding layers.")
parser.add_argument('--num-hidden', type=int, default=1,
                    help="num-hidden of neumf fc layer")
parser.add_argument('--gpus', type=str,
                    help="list of gpus to run, e.g. 0 or 0,2. empty means using cpu().")
parser.add_argument('--sparse', action='store_true', help="whether to use sparse embedding")
parser.add_argument('--ckt-index', type=int, default=2, help='model checkpoint index for inference')


if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)

    batch_size = args.batch_size
    model_type = args.model_type
    factor_size_mlp = args.factor_size_mlp
    factor_size_gmf = args.factor_size_gmf
    model_layers = args.layers
    num_hidden = args.num_hidden
    sparse = args.sparse
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus else [mx.cpu()]
    topK = 10
    evaluation_threads = 1#mp.cpu_count()

    # prepare dataset and iterators
    logging.info('Prepare Dataset')
    train_iter, val_iter, max_user, max_movies = get_dataset(args.path, num_train = 19000000, batch_size = batch_size)
    logging.info('Prepare Dataset completed')
    # construct the model
    net = get_model(model_type, factor_size_mlp, factor_size_gmf, 
                    model_layers, num_hidden, max_user, max_movies, sparse)
    net = net.get_backend_symbol("MKLDNN")
    # initialize the module
    mod = mx.module.Module(net, context=ctx, data_names=['user', 'item'], label_names=['softmax_label'])
    mod.bind(for_training=False, data_shapes=val_iter.provide_data, label_shapes=val_iter.provide_label)
    mod.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
    logging.info('Inference...')
    tic = time.time()
    num_samples = 0
    for batch in val_iter:
        mod.forward(batch, is_train=False)
        num_samples += batch_size
    toc = time.time()
    fps = num_samples/(toc - tic)
    logging.info('Evaluating completed')
    logging.info('Inference speed %.4f fps' % fps)

    # (hits, ndcgs) = evaluate_model(mod, testRatings, testNegatives, topK, evaluation_threads)
    # hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    # print('Evaluate: HR = %.4f, NDCG = %.4f'  % (hr, ndcg))

