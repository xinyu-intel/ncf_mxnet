import mxnet as mx
from core.model import get_model

def test_model():
    def test_ncf(model_type):
        net = get_model(model_type=model_type, factor_size_mlp=128, factor_size_gmf=64, 
                        model_layers=[256, 128, 64], num_hidden=1, max_user=138493, max_item=26744)
        mod = mx.module.Module(net, context=mx.cpu(), data_names=['user', 'item'], label_names=['softmax_label'])
        provide_data = [mx.io.DataDesc(name='item', shape=((1,))),
                        mx.io.DataDesc(name='user', shape=((1,)))]
        provide_label = [mx.io.DataDesc(name='softmax_label', shape=((1,)))]
        mod.bind(for_training=False, data_shapes=provide_data, label_shapes=provide_label)
        mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
        data = [mx.nd.full(shape=shape, val=127, ctx=mx.cpu(), dtype='uint8')
                for _, shape in mod.data_shapes]
        batch = mx.io.DataBatch(data, [])
        mod.forward(batch)
        mx.nd.waitall()
    for model_type in ['neumf', 'mlp', 'gmf']:
        test_ncf(model_type)

if __name__ == "__main__":
    import nose
    nose.runmodule()

