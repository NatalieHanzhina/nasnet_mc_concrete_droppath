import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.python.framework.smart_cond import smart_cond


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
            # tf.print(input.shape, tf.reduce_sum(binary_tensor))
            # tf.print(input.shape)
            tf.Assert(tf.convert_to_tensor(
                tf.reduce_sum(binary_tensor)) >=
                      tf.convert_to_tensor(tf.reduce_sum(
                          tf.cast(tf.reduce_max(outputs, axis=[1,2,3]) > 0, dtype=tf.float32))),
                      data=['Nothing'])  # TODO: remove when debugged
            #tf.print(self.name + ': \t', tf.reduce_sum(binary_tensor), tf.shape(binary_tensor)[0], end='')
            #tf.print(';\t', tf.reduce_sum(tf.cast(tf.reduce_max(outputs, axis=[1,2,3]) > 0, dtype=tf.int16)))
            return outputs

        output = smart_cond(training, dropped_inputs, lambda: tf.identity(inputs))

        # tf.print('Maxes before droppath', [tf.math.reduce_max(inp_path) for inp_path in inputs])
        # tf.print('Maxes after droppath', [tf.math.reduce_max(out_path) for out_path in output])
        return output

    def _compute_scheduled_dropout_rate(self):
        drop_rate = self.drop_rate
        # assert drop_connect_version in ['v1', 'v2', 'v3']
        if self._total_num_cells is not None:  #drop_connect_version in ['v2', 'v3']:
            # Scale keep prob by layer number
            assert self._cell_num != -1
            # The added 2 is for the reduction cells
            num_cells = self._total_num_cells
            layer_ratio = (self._cell_num + 1) / float(num_cells)
            # if use_summaries:
            #     with tf.device('/cpu:0'):
            #         tf.summary.scalar('layer_ratio', layer_ratio)
            drop_rate = layer_ratio * drop_rate
        if self._total_training_steps is not None:  #drop_connect_version in ['v1', 'v3']:
            # Decrease the keep probability over time
            # if current_step is None:
            #     current_step = tf.compat.v1.train.get_or_create_global_step()
            current_step = tf.convert_to_tensor(tf.compat.v1.train.get_or_create_global_step())
            tf.compat.v1.get_variable_scope().reuse_variables()
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
            'cell_num': self._cell_num,
            'total_num_cells': self._total_num_cells,
            'total_training_steps': self._total_training_steps,
            'seed': self.seed
        }
        base_config = super(ScheduledDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConcreteDroppath(Layer):
    """Applies Concrete Dropout to the input.
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

    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDroppath, self).__init__(**kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = tf.math.log(init_min) - tf.math.log(1. - init_min)
        self.init_max = tf.math.log(init_max) - tf.math.log(1. - init_max)

    @tf.function
    def get_p(self):
        return tf.nn.sigmoid(self.p_logit[0])


    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        # if not self.layer.built:
        #     self.layer.build(input_shape)
        #     self.layer.built = True
        super(ConcreteDroppath, self).build(input_shape)  # this is very weird.. we must call super before we add new losses

        # initialise p
        #self.p_logit = self.layer.add_weight(name='p_logit',
        self.p_logit = self.add_weight(name='p_logit',
                                             shape=(1,),
                                             initializer=tf.initializers.RandomUniform(self.init_min, self.init_max),
                                             trainable=True)
        #self.p = K.sigmoid(self.p_logit[0])
        #self.p = self.add_weight(name='p', trainable=False)

        # initialise regulariser / prior KL term
        #assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        #input_dim = np.prod(input_shape[-1])  # we drop only last dim
        #weight = self.layer.kernel
        #kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.get_p() * K.log(self.get_p())
        dropout_regularizer += (1. - self.get_p()) * K.log(1. - self.get_p())
        #dropout_regularizer *= self.dropout_regularizer * input_dim
        dropout_regularizer *= self.dropout_regularizer
        #regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        regularizer = dropout_regularizer
        #self.layer.add_loss(regularizer)
        self.add_loss(regularizer)

    # def call(self, inputs, training=None):
    #     if self.is_mc_dropout:
    #         return self.layer.call(self.concrete_dropout(inputs))
    #     else:
    #         def relaxed_dropped_inputs():
    #             return self.layer.call(self.concrete_dropout(inputs))
    #         return K.in_train_phase(relaxed_dropped_inputs,
    #                                 self.layer.call(inputs),
    #                                 training=training)

    def call(self, inputs, training=None):
        #tf.print(self.p_logit)
        if self.is_mc_dropout:
            return self.concrete_dropout(inputs)
        else:
            def relaxed_dropped_inputs():
                return self.concrete_dropout(inputs)
            return K.in_train_phase(relaxed_dropped_inputs,
                                    inputs,
                                    training=training)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''

        x_init = tf.identity(x)
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        #tf.print('x_in', tf.reduce_min(x), tf.reduce_max(x))
        #tf.print(self.get_p())
        #tf.print(type(K.shape(x)))
        #tf.print(tf.concat([K.shape(x)[0:1], tf.ones((3,), dtype=K.shape(x).dtype)], axis=0))
        #tf.print(K.shape(x)[0])
        unif_noise = K.random_uniform(shape=tf.concat([K.shape(x)[0:1], tf.ones((3,), dtype=K.shape(x).dtype)], axis=0))
        drop_prob = (
            K.log(self.get_p() + eps)
            - K.log(1. - self.get_p() + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.get_p()
        #tf.print(self.name, ':', K.shape(random_tensor), K.shape(retain_prob))
        #tf.print(self.name, ':\n', (1* random_tensor / retain_prob)[:,0,0,0], '\n', summarize=30)
        #tf.print(self.name, ':\n', self.get_p(), '\n', summarize=30)
        #tf.print(self.name, ':\n', (1 * random_tensor / retain_prob)[:,0,0,0], summarize=30)
        x *= random_tensor
        x /= retain_prob
        #tf.print('x_out', tf.reduce_min(x), tf.reduce_max(x))
        #tf.print(self.name, ':\n', tf.reduce_mean(x-x_init, axis=(1,2,3)), '\n', summarize=30)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self): #TODO: refactor
        config = {
            # 'drop_rate': self.drop_rate,
            # 'cell_num': self._cell_num,
            # 'total_num_cells': self._total_num_cells,
            # 'total_training_steps': self._total_training_steps,
            # 'seed': self.seed
        }
        base_config = super(ConcreteDroppath, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
