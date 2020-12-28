from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, LayerNormalization
from vit_keras.layers.transformer import TransformerEncoderLayer, TransformerInputConv2DLayer
from tensorflow.io import gfile
import numpy as np

class VitBuilder():
    def __init__(self, image_size=(384, 384, 3), patch_size=16, num_heads=12, num_layers=12, num_classes=1000, load_pretrain=True):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.load_pretrain = load_pretrain

    def build(self):
        input_layer = Input(shape=(384, 384, 3), batch_size=1)
        x = TransformerInputConv2DLayer(self.image_size, self.patch_size)(input_layer)
        for i in range(self.num_layers):
            x = TransformerEncoderLayer(f'Transformer/encoderblock_{i}', self.image_size, self.patch_size, self.num_heads)(x)
        x = LayerNormalization(name='encoder_norm')(x)
        output = Dense(self.num_classes, name='head')(x[:, 0])

        vit_model = keras.Model(input_layer, output)

        if self.load_pretrain:
            self.load_pretrain_weights(vit_model)

        return vit_model

    def load_pretrain_weights(self, vit):
        with gfile.GFile('ViT-B_16_imagenet2012.npz', 'rb') as f:
            ckpt_dict = np.load(f, allow_pickle=False)

        for layer in vit.get_layer('Transformer/posembed_input').trainable_weights:
            if 'cls:0' == layer.name:
                layer.assign(ckpt_dict['cls'])
            if 'position_embedding:0' == layer.name:
                layer.assign(ckpt_dict['Transformer/posembed_input/pos_embedding'])
            if 'Transformer/posembed_input/embedding/kernel:0' == layer.name:
                layer.assign(ckpt_dict['embedding/kernel'].reshape(layer.shape))
            if 'Transformer/posembed_input/embedding/bias:0' == layer.name:
                layer.assign(ckpt_dict['embedding/bias'])

        for i in range(self.num_layers):
            for layer in vit.get_layer(f'Transformer/encoderblock_{i}').trainable_weights:
                for k in [0, 2]:
                    if layer.name.endswith(f'/LayerNorm_{k}/gamma:0'):
                        layer.assign(ckpt_dict[f'Transformer/encoderblock_{i}/LayerNorm_{k}/scale'])
                    if layer.name.endswith(f'/LayerNorm_{k}/beta:0'):
                        layer.assign(ckpt_dict[f'Transformer/encoderblock_{i}/LayerNorm_{k}/bias'])

                for qkv in ['query', 'key', 'value', 'out']:
                    if layer.name.endswith(f'/{qkv}_kernel:0'):
                        layer.assign(
                            ckpt_dict[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/{qkv}/kernel'])
                    if layer.name.endswith(f'/{qkv}_bias:0'):
                        layer.assign(
                            ckpt_dict[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/{qkv}/bias'])

                for k in range(2):
                    if layer.name.endswith(f'/Dense_{k}/kernel:0'):
                        layer.assign(ckpt_dict[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_{k}/kernel'])
                    if layer.name.endswith(f'/Dense_{k}/bias:0'):
                        layer.assign(ckpt_dict[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_{k}/bias'])

        for layer in vit.get_layer('encoder_norm').trainable_weights:
            if 'encoder_norm/gamma:0' == layer.name:
                layer.assign(ckpt_dict['Transformer/encoder_norm/scale'])
            if 'encoder_norm/beta:0' == layer.name:
                layer.assign(ckpt_dict['Transformer/encoder_norm/bias'])

        for layer in vit.get_layer('head').trainable_weights:
            if 'head/kernel:0' == layer.name:
                layer.assign(ckpt_dict['head/kernel'])
            if 'head/bias:0' == layer.name:
                layer.assign(ckpt_dict['head/bias'])
