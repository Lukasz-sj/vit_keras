from tensorflow.keras.layers import LayerNormalization, Dropout
from layers import MultiHeadAttention, MLPLayer


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
