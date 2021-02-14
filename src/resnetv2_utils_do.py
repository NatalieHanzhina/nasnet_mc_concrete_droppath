import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils.tf_utils import smart_cond


class DropPath(Layer):
    """Applies Droppath to the input.
        The Dropout layer randomly sets input units to 0 with a frequency of `rate`
        at each step during training time, which helps prevent overfitting.
        Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
        all inputs is unchanged.
        Note that the Dropout layer only applies when `training` is set to True
        such that no values are dropped during inference. When using `model.fit`,
        `training` will be appropriately set to True automatically, and in other
        contexts, you can set the kwarg explicitly to True when calling the layer.
        (This is in contrast to setting `trainable=False` for a Dropout layer.
        `trainable` does not affect the layer's behavior, as Dropout does
        not have any variables/weights that can be frozen during training.)
        Arguments:
            paths_rate: Float between 0 and 1. Fraction of the input units to drop.
            noise_shape: 1D integer tensor representing the shape of the
                binary dropout mask that will be multiplied with the input.
                For instance, if your inputs have shape
                `(batch_size, timesteps, features)` and
                you want the dropout mask to be the same for all timesteps,
                you can use `noise_shape=(batch_size, 1, features)`.
            seed: A Python integer to use as random seed.
        Call arguments:
            inputs: Input tensor (of any rank).
            training: Python boolean indicating whether the layer should behave in
                training mode (adding dropout) or in inference mode (doing nothing).
    """

    def __init__(self, paths_rate, drop_paths_mask, seed=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_rate = paths_rate
        self.drop_paths_mask = drop_paths_mask
        self.seed = seed
        self.supports_masking = True

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        drop_paths_count = np.sum(self.drop_paths_mask)
        drop_paths_flags = np.random.choice([0, 1], drop_paths_count, p=[self.drop_rate, 1-self.drop_rate])
        selected_indicies = np.where(drop_paths_flags == 0)[0]
        # tf.print('drop_indices:', tf.convert_to_tensor(selected_indicies), 'drop_len:', int(drop_paths_count*self.paths_rate))
        if len(selected_indicies) == drop_paths_count:
            selected_indicies = np.random.choice(drop_paths_count, drop_paths_count - 1)
        drop_paths_counter = -1
        output = []

        for i, inp in enumerate(inputs):
            if self.drop_paths_mask[i]:
                drop_paths_counter += 1
            if training:
                output.append(smart_cond(drop_paths_counter in selected_indicies,
                                         lambda: tf.math.multiply(inp, 0),
                                         lambda: tf.math.multiply(inp, 1.0 / (1 - self.drop_rate))))
            else:
                output.append(tf.identity(inp))

        for i in selected_indicies:
            tf.debugging.Assert(tf.math.reduce_max(output[i]) == 0., data=[tf.reduce_max(out) for out in output],
                                summarize=3)
        # for i, br in enumerate(output):
        #     tf.print(f'branch {i}: max', tf.math.reduce_max(br))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'drop_rate': self.drop_rate,
            'drop_paths_mask': self.drop_paths_mask,
            'seed': self.seed
        }
        base_config = super(DropPath, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
