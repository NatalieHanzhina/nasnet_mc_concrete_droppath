import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils.tf_utils import smart_cond


class ScheduledDropout(Layer):
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
            drop_rate: Float between 0 and 1. Fraction of the input units to drop.
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
    """  #TODO: refactor docs

    def __init__(self, drop_rate, cell_num, total_num_cells, total_training_steps, seed=None, **kwargs):
        super(ScheduledDropout, self).__init__(**kwargs)
        self.drop_rate = drop_rate
        self._cell_num = cell_num
        self._total_num_cells = total_num_cells
        self._total_training_steps = total_training_steps
        self.seed = seed

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        scheduled_drop_rate = self._compute_scheduled_dropout_rate()

        def dropped_inputs():
            noise_shape = [tf.shape(input=inputs)[0], 1, 1, 1]
            # noise_shape = [1, 1, 1, 1]
            random_tensor = 1 - scheduled_drop_rate
            random_tensor += tf.random.uniform(noise_shape, dtype=tf.float32)
            binary_tensor = tf.cast(tf.floor(random_tensor), inputs.dtype)
            keep_prob_inv = tf.cast(1.0 / (1-scheduled_drop_rate), inputs.dtype)
            outputs = inputs * keep_prob_inv * binary_tensor
            return outputs

        output = smart_cond(training, dropped_inputs, lambda: tf.identity(inputs))

        # tf.print('Maxes before droppath', [tf.math.reduce_max(inp_path) for inp_path in inputs])
        # tf.print('Maxes after droppath', [tf.math.reduce_max(out_path) for out_path in output])
        return output

    def _compute_scheduled_dropout_rate(self):
        drop_rate = self.drop_rate
        # assert drop_connect_version in ['v1', 'v2', 'v3']
        if True:  #drop_connect_version in ['v2', 'v3']:
            # Scale keep prob by layer number
            assert self._cell_num != -1
            # The added 2 is for the reduction cells
            num_cells = self._total_num_cells
            layer_ratio = (self._cell_num + 1) / float(num_cells)
            # if use_summaries:
            #     with tf.device('/cpu:0'):
            #         tf.summary.scalar('layer_ratio', layer_ratio)
            drop_rate = layer_ratio * drop_rate
        if True:  #drop_connect_version in ['v1', 'v3']:
            # Decrease the keep probability over time
            # if current_step is None:
            #     current_step = tf.compat.v1.train.get_or_create_global_step()
            current_step = tf.convert_to_tensor(tf.compat.v1.train.get_or_create_global_step())
            current_step = tf.cast(current_step, tf.float32)
            drop_path_burn_in_steps = self._total_training_steps
            current_ratio = current_step / drop_path_burn_in_steps
            current_ratio = tf.minimum(1.0, current_ratio)
            # if use_summaries:
            #     with tf.device('/cpu:0'):
            #         tf.summary.scalar('current_ratio', current_ratio)
            drop_rate = current_ratio * drop_rate
        # if use_summaries:
        #     with tf.device('/cpu:0'):
        #         tf.summary.scalar('drop_path_keep_prob', drop_path_keep_prob)
        return drop_rate

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'drop_rate': self.drop_rate,
            'drop_paths_mask': self.drop_paths_mask,
            'seed': self.seed
        }
        base_config = super(ScheduledDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
