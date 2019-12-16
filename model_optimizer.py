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
import math
import random
import numpy as np
import mxnet as mx
from core.model import get_model
from core.dataset import NCFTrainData

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Run model optimizer.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', nargs='?', default='./data/',
                    help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='ml-20m',
                    help='The dataset name.')
parser.add_argument('--model_parameters', type=str, default='./model/ml-20m/neumf-0007.params')
parser.add_argument('--model-type', type=str, default='neumf', choices=['neumf', 'gmf', 'mlp'],
                    help="mdoel type")
parser.add_argument('--layers', default='[256, 256, 128, 64]',
                    help="list of number hiddens of fc layers in mlp model.")
parser.add_argument('--factor-size-gmf', type=int, default=64,
                    help="outdim of gmf embedding layers.")
parser.add_argument('--num-hidden', type=int, default=1,
                    help="num-hidden of neumf fc layer")

head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=head)

# arg parser
args = parser.parse_args()
logging.info(args)

model_parameters = args.model_parameters
model_type = args.model_type
model_layers = eval(args.layers)
factor_size_gmf = args.factor_size_gmf
factor_size_mlp = int(model_layers[0]/2)
num_hidden = args.num_hidden
#train_dataset = NCFTrainData((args.path + args.dataset + '/train-ratings.csv'), nb_neg=4)
#net = get_model(model_type, factor_size_mlp, factor_size_gmf, 
#                model_layers, num_hidden, train_dataset.nb_users, train_dataset.nb_items, opt=True)
net = get_model(model_type, factor_size_mlp, factor_size_gmf, 
                model_layers, num_hidden, 138493, 26744,  opt=True)

raw_params = mx.nd.load(model_parameters)
fc_0_weight_split = mx.nd.split(raw_params['arg:fc_0_weight'], axis=1, num_outputs=2)
fc_0_left = fc_0_weight_split[0]
fc_0_right = fc_0_weight_split[1]

user_weight_fusion = mx.nd.FullyConnected(data = raw_params['arg:mlp_user_weight'], weight=fc_0_left, bias=raw_params['arg:fc_0_bias'], no_bias=False, num_hidden=model_layers[0])
item_weight_fusion = mx.nd.FullyConnected(data = raw_params['arg:mlp_item_weight'], weight=fc_0_right, no_bias=True, num_hidden=model_layers[0])

#user = mx.nd.random.randint(low = 1, high = 138493, shape = (1000,))
#item = mx.nd.random.randint(low = 1, high = 26744, shape = (1000,))

#embed_user = mx.nd.Embedding(data=user, weight=raw_params['arg:mlp_user_weight'], sparse_grad=False, input_dim=138493,output_dim=128)
#embed_item = mx.nd.Embedding(data=item, weight=raw_params['arg:mlp_item_weight'], sparse_grad=False, input_dim=26744,output_dim=128)
#pre_gemm_concat = mx.nd.concat(embed_user, embed_item, dim=1, name='pre_gemm_concat')
#raw_net = mx.nd.FullyConnected(data=pre_gemm_concat, weight=raw_params['arg:fc_0_weight'], bias=raw_params['arg:fc_0_bias'], num_hidden=256)
#print(raw_net)


#embed_user_fusion = mx.nd.Embedding(data=user, weight=user_weight_fusion, sparse_grad=False, input_dim=138493,output_dim=256)
#embed_item_fusion = mx.nd.Embedding(data=item, weight=item_weight_fusion, sparse_grad=False, input_dim=26744,output_dim=256)
#print(embed_user_fusion + embed_item_fusion)

opt_params = {}
opt_params['arg:gmf_user_weight'] = raw_params['arg:gmf_user_weight']
opt_params['arg:gmf_item_weight'] = raw_params['arg:gmf_item_weight']
opt_params['arg:fused_mlp_user_weight'] = user_weight_fusion
opt_params['arg:fused_mlp_item_weight'] = item_weight_fusion
opt_params['arg:fc_1_weight'] = raw_params['arg:fc_1_weight']
opt_params['arg:fc_1_bias'] = raw_params['arg:fc_1_bias']
opt_params['arg:fc_2_weight'] = raw_params['arg:fc_2_weight']
opt_params['arg:fc_2_bias'] = raw_params['arg:fc_2_bias']
opt_params['arg:fc_final_weight'] = raw_params['arg:fc_final_weight']
opt_params['arg:fc_final_bias'] = raw_params['arg:fc_final_bias']

mx.nd.save('neurf-opt-0000.params', opt_params)
net.save('neurf-opt-symbol.json')
