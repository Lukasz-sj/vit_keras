from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, LayerNormalization
from layers import TransformerEncoderLayer, TransformerInputConv2DLayer

class VitBuilder():
    def __init__(self, image_size=(384, 384, 3), patch_size=16, num_heads=12, num_layers=12, num_classes=1000):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

    def build(self):
        input_layer = Input(shape=(384, 384, 3), batch_size=1)
        x = TransformerInputConv2DLayer(self.image_size, self.patch_size)(input_layer)
        for i in range(self.num_layers):
            x = TransformerEncoderLayer(f'Transformer/encoderblock_{i}', self.image_size, self.patch_size, self.num_heads)(x)
        x = LayerNormalization(name='encoder_norm')(x)
        output = Dense(self.num_classes, name='head')(x[:, 0])

        return keras.Model(input_layer, output)
