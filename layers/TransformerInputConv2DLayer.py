import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D


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
