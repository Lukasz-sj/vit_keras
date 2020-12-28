import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, Conv2D
from tensorflow.keras.layers import LayerNormalization
from vit_keras.layers.tensorflow_addon import MultiHeadAttention


class MLPLayer(keras.layers.Layer):

    def __init__(self, image_size, patch_size):
        super(MLPLayer, self).__init__()
        p = patch_size
        c = image_size[2]
        embeded_dim = c * p ** 2
        self.layer1 = Dense(4 * embeded_dim, activation=tfa.activations.gelu, name='Dense_0')
        self.dropout1 = Dropout(0.1)
        self.layer2 = Dense(embeded_dim, name='Dense_1')
        self.dropout2 = Dropout(0.1)

    def call(self, x):
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        return self.dropout2(x)


class TransformerInputConv2DLayer(keras.layers.Layer):

    def __init__(self, image_size=None, patch_size=None):
        super(TransformerInputConv2DLayer, self).__init__(name='Transformer/posembed_input')
        self.p = patch_size
        self.h = image_size[0]
        self.w = image_size[1]
        self.c = image_size[2]
        self.n = (int)(self.h * self.w / self.p ** 2)
        self.embeded_dim = self.c * self.p ** 2

        self.class_embedding = self.add_weight("cls", shape=(1, 1, self.embeded_dim), trainable=True)
        self.position_embedding = self.add_weight("position_embedding", shape=(1, self.n + 1, self.embeded_dim),
                                                  trainable=True)
        self.linear_projection = Conv2D(self.embeded_dim, self.p, strides=(self.p, self.p), padding='valid',
                                        name='embedding')
        self.dropout = Dropout(0.1)

    def call(self, x):
        batch_size = x.shape[0]

        if batch_size is None:
            batch_size = -1

        x = self.linear_projection(x)
        n, h, w, c = x.shape
        reshaped_image_patches = tf.reshape(x, [n, h * w, c])

        class_embedding = tf.broadcast_to(self.class_embedding, [batch_size, 1, self.embeded_dim])
        reshaped_image_patches = tf.concat([class_embedding, reshaped_image_patches], axis=1)
        reshaped_image_patches += self.position_embedding
        reshaped_image_patches = self.dropout(reshaped_image_patches)
        x = reshaped_image_patches

        return x


class TransformerEncoderLayer(keras.layers.Layer):

    def __init__(self, name, image_size, patch_size, num_heads):
        super(TransformerEncoderLayer, self).__init__(name=name)
        p = patch_size
        c = image_size[2]
        self.embeded_dim = c * p ** 2
        head_size = (int)(self.embeded_dim / num_heads)
        self.layer_normalization1 = LayerNormalization(name='LayerNorm_0')
        self.multi_head_attention = MultiHeadAttention(head_size, num_heads)
        self.dropout = Dropout(0.1)
        self.layer_normalization2 = LayerNormalization(name='LayerNorm_2')
        self.mlp_layer = MLPLayer(image_size, patch_size)

    def call(self, x):
        input_x = x
        x = self.layer_normalization1(x)
        x = self.multi_head_attention([x, x])
        x = self.dropout(x)
        x = x + input_x
        y = self.layer_normalization2(x)
        y = self.mlp_layer(y)
        return x + y