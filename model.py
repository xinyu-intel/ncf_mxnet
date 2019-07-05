import mxnet as mx

def mlp(user, item, factor_size, model_layers, max_user, max_item, sparse):
    stype = 'row_sparse' if sparse else 'default'
    user_weight = mx.sym.Variable('mlp_user_weight', stype=stype, init=mx.init.Normal(0.01))
    item_weight = mx.sym.Variable('mlp_item_weight', stype=stype, init=mx.init.Normal(0.01))
    embed_user = mx.sym.Embedding(data=user, weight=user_weight, sparse_grad=sparse, input_dim=max_user,
                                  output_dim=factor_size, name='embed_user'+str(factor_size))
    embed_item = mx.sym.Embedding(data=item, weight=item_weight, sparse_grad=sparse, input_dim=max_item,
                                  output_dim=factor_size, name='embed_item'+str(factor_size))
    pre_gemm_concat = mx.sym.concat(embed_user, embed_item, dim=1, name='pre_gemm_concat')
    for i, layer in enumerate(model_layers):
        pre_gemm_concat = mx.sym.FullyConnected(data=pre_gemm_concat, num_hidden=layer, name='fc_'+str(i))
        pre_gemm_concat = mx.sym.Activation(data=pre_gemm_concat, act_type='relu', name='act_'+str(i))

    return pre_gemm_concat

def gmf(user, item, factor_size, max_user, max_item, sparse):
    stype = 'row_sparse' if sparse else 'default'
    user_weight = mx.sym.Variable('gmf_user_weight', stype=stype, init=mx.init.Normal(0.01))
    item_weight = mx.sym.Variable('gmf_item_weight', stype=stype, init=mx.init.Normal(0.01))
    embed_user = mx.sym.Embedding(data=user, weight=user_weight, sparse_grad=sparse, input_dim=max_user,
                                  output_dim=factor_size, name='embed_user'+str(factor_size))
    embed_item = mx.sym.Embedding(data=item, weight=item_weight, sparse_grad=sparse, input_dim=max_item,
                                  output_dim=factor_size, name='embed_item'+str(factor_size))
    pred = embed_user * embed_item

    return pred

def get_model(model_type='ncf', factor_size_mlp=128, factor_size_gmf=64,
              model_layers=[256, 128, 64], num_hidden=1, 
              max_user=138493, max_item=26744, sparse=False):
    # input
    user = mx.sym.Variable('user')
    item = mx.sym.Variable('item')

    if model_type == 'mlp':
        net = mlp(user=user, item=item,
                  factor_size=factor_size_mlp, model_layers=model_layers,
                  max_user=max_user, max_item=max_item, sparse=sparse)
    elif model_type == 'gmf':
        net = gmf(user=user, item=item,
                  factor_size=factor_size_gmf,
                  max_user=max_user, max_item=max_item, sparse=sparse)
    elif model_type == 'neumf':
        net_mlp = mlp(user=user, item=item,
                      factor_size=factor_size_mlp, model_layers=model_layers,
                      max_user=max_user, max_item=max_item, sparse=sparse)
        net_gmf = gmf(user=user, item=item,
                      factor_size=factor_size_gmf,
                      max_user=max_user, max_item=max_item, sparse=sparse)

        net = mx.sym.concat(net_gmf, net_mlp, name='post_gemm_concat')

    else:
        raise ValueError('Unsupported ncf model %s.' % model_type)

    net = mx.sym.FullyConnected(data=net, num_hidden=num_hidden, name='fc_final') 
    y_label = mx.sym.Variable('softmax_label')
    net = mx.symbol.LogisticRegressionOutput(data=net, label=y_label, name='sigmoid_final')

    return net
