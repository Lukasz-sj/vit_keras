from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Layer
import tensorflow_addons as tfa


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
