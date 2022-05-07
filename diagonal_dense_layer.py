import tensorflow as tf
from keras import layers


class DiagonalDense(layers.Dense):
    def __init__(self ,
                 units,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
        super(DiagonalDense, self).__init__(
            units, None, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, **kwargs)


    def call(self, inputs):
        if inputs.shape[-2] != self.units:
            raise ValueError('DiagonalDense layer requires the second to last dimension of inputs to be equal to the number of units.'
                             f' Received: inputs.shape={inputs.shape}, units={self.units}')
        rank = inputs.shape.rank
        #no need to do anything
        if rank == 2 or rank is None:
            return super(DiagonalDense,self).call(inputs)
        elif rank == 3:
            einsum_str = 'kii->ki'
        elif rank == 4:
            einsum_str = 'mkii->mki'
        else:
            raise ValueError('DiagonalDense layer does not support inputs with rank greater than 4.')

        return tf.einsum( einsum_str, super(DiagonalDense,self).call(inputs))

    #override the parent method the output shape now is simply the input shape without the last dimension
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
          raise ValueError('The last dimension of the input shape of a Dense layer '
                           'should be defined. Found None. '
                           f'Received: input_shape={input_shape}')
        return input_shape[:-1]
