import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


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

    def __init__(self, paths_rate, drop_paths, seed=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.paths_rate = paths_rate
        self.drop_paths = drop_paths
        self.seed = seed
        self.supports_masking = True

    # def _get_noise_shape(self, inputs):
    #     # Subclasses of `Dropout` may implement `_get_noise_shape(self, inputs)`,
    #     # which will override `self.noise_shape`, and allows for custom noise
    #     # shapes with dynamically sized inputs.
    #     if self.noise_shape is None:
    #         return None
    #
    #     concrete_inputs_shape = array_ops.shape(inputs)
    #     noise_shape = []
    #     for i, value in enumerate(self.noise_shape):
    #         noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    #     return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        drop_paths_count = np.sum(self.drop_paths)
        selected_incdicies = np.random.choice(drop_paths_count, int(drop_paths_count*self.paths_rate))
        if len(selected_incdicies) == drop_paths_count:
            selected_incdicies = np.random.choice(drop_paths_count, drop_paths_count-1)
        drop_paths_counter = -1
        output = []

        def dropped_inputs(input_to_drop):
            return tf.nn.dropout(
                input_to_drop,
                noise_shape=tf.convert_to_tensor([1,]*input_to_drop.shape),
                seed=self.seed,
                rate=1)
        for i, inp in enumerate(inputs):
            if self.drop_paths[i]:
                drop_paths_counter += 1
            output.append(tf.cond(training and drop_paths_counter in selected_incdicies, lambda: dropped_inputs(inp),
                                  lambda: tf.identity(inp)))
            # output.append(smart_cond(training and drop_paths_counter in selected_incdicies, lambda: tf.matmul(inp, 0),
            #                          lambda: tf.identity(inp)))

        # output = control_flow_util.smart_cond(training, dropped_inputs,
        #                                       lambda: tf.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        }
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
