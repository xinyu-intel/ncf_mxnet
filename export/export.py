import numpy as np
import torch
import torch.nn as nn
import mxnet as mx

class NeuMF(nn.Module):
    def __init__(self, nb_users, nb_items,
                 mf_dim, mf_reg,
                 mlp_layer_sizes, mlp_layer_regs):
        if len(mlp_layer_sizes) != len(mlp_layer_regs):
            raise RuntimeError('u dummy, layer_sizes != layer_regs!')
        if mlp_layer_sizes[0] % 2 != 0:
            raise RuntimeError('u dummy, mlp_layer_sizes[0] % 2 != 0')
        super(NeuMF, self).__init__()
        nb_mlp_layers = len(mlp_layer_sizes)

        # TODO: regularization?
        self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
        self.mf_item_embed = nn.Embedding(nb_items, mf_dim)
        self.mlp_user_embed = nn.Embedding(nb_users, mlp_layer_sizes[0] // 2)
        self.mlp_item_embed = nn.Embedding(nb_items, mlp_layer_sizes[0] // 2)

        self.mlp = nn.ModuleList()
        for i in range(1, nb_mlp_layers):
            self.mlp.extend([nn.Linear(mlp_layer_sizes[i - 1], mlp_layer_sizes[i])])  # noqa: E501

        self.final = nn.Linear(mlp_layer_sizes[-1] + mf_dim, 1)

        self.mf_user_embed.weight.data.normal_(0., 0.01)
        self.mf_item_embed.weight.data.normal_(0., 0.01)
        self.mlp_user_embed.weight.data.normal_(0., 0.01)
        self.mlp_item_embed.weight.data.normal_(0., 0.01)

        def golorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)
        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            golorot_uniform(layer)
        lecunn_uniform(self.final)

    def forward(self, user, item, sigmoid=False):
        xmfu = self.mf_user_embed(user)
        xmfi = self.mf_item_embed(item)
        xmf = xmfu * xmfi

        xmlpu = self.mlp_user_embed(user)
        xmlpi = self.mlp_item_embed(item)
        xmlp = torch.cat((xmlpu, xmlpi), dim=1)
        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp)
            xmlp = nn.functional.relu(xmlp)

        x = torch.cat((xmf, xmlp), dim=1)
        x = self.final(x)
        if sigmoid:
            x = nn.functional.sigmoid(x)
        return x

model = NeuMF(138493, 26744,
              mf_dim=64, mf_reg=0.,
              mlp_layer_sizes=[256, 256, 128, 64],
              mlp_layer_regs=[0, 0, 0, 0])

model.load_state_dict(torch.load('./neumf_model_6'))
mf_user_embed = model.mf_user_embed.weight.data.numpy()
mf_item_embed = model.mf_item_embed.weight.data.numpy()
mlp_user_embed = model.mlp_user_embed.weight.data.numpy()
mlp_item_embed = model.mlp_item_embed.weight.data.numpy()
mlp0 = model.mlp[0].weight.data.numpy()
mlp0_bias = model.mlp[0].bias.data.numpy()
mlp1 = model.mlp[1].weight.data.numpy()
mlp1_bias = model.mlp[1].bias.data.numpy()
mlp2 = model.mlp[2].weight.data.numpy()
mlp2_bias = model.mlp[2].bias.data.numpy()
final = model.final.weight.data.numpy()
final_bias = model.final.bias.data.numpy()

arg_params = {}
aux_params = {}
arg_params['gmf_user_weight'] = mx.nd.array(mf_user_embed)
arg_params['gmf_item_weight'] = mx.nd.array(mf_item_embed)
arg_params['mlp_user_weight'] = mx.nd.array(mlp_user_embed)
arg_params['mlp_item_weight'] = mx.nd.array(mlp_item_embed)
arg_params['fc_0_weight'] = mx.nd.array(mlp0)
arg_params['fc_0_bias'] = mx.nd.array(mlp0_bias)
arg_params['fc_1_weight'] = mx.nd.array(mlp1)
arg_params['fc_1_bias'] = mx.nd.array(mlp1_bias)
arg_params['fc_2_weight'] = mx.nd.array(mlp2)
arg_params['fc_2_bias'] = mx.nd.array(mlp2_bias)
arg_params['fc_final_weight'] = mx.nd.array(final)
arg_params['fc_final_bias'] = mx.nd.array(final_bias)
save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
mx.nd.save('neumf-0000.params', save_dict)
