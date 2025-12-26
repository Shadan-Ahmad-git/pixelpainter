import tensorflow as tf

# Custom objects for loading
class LegacyTruncatedNormal(tf.keras.initializers.TruncatedNormal):
    def __init__(self, mean=0.0, stddev=0.05, seed=None, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__(mean=mean, stddev=stddev, seed=seed)

class LegacyZeros(tf.keras.initializers.Zeros):
    def __init__(self, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__()

class LegacyOnes(tf.keras.initializers.Ones):
    def __init__(self, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__()

class LegacyRandomNormal(tf.keras.initializers.RandomNormal):
    def __init__(self, mean=0.0, stddev=0.05, seed=None, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__(mean=mean, stddev=stddev, seed=seed)

class TensorFlowOpLayer(tf.keras.layers.Layer):
    def __init__(self, node_def, name=None, **kwargs):
        super().__init__(name=name)
        self.node_def = node_def
    def call(self, inputs):
        return inputs

custom_objects = {
    'TruncatedNormal': LegacyTruncatedNormal,
    'Zeros': LegacyZeros,
    'Ones': LegacyOnes,
    'RandomNormal': LegacyRandomNormal,
    'TensorFlowOpLayer': TensorFlowOpLayer,
    'tf': tf
}

print("Loading model from checkpoints/generator_weights.h5...")
model = tf.keras.models.load_model('checkpoints/generator_weights.h5', custom_objects=custom_objects, compile=False)
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
