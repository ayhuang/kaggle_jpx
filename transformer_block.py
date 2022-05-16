import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=128, #embed_dim,
                                             kernel_initializer="glorot_uniform",
                                            # attention_axes = (1,2),
                                             dropout=rate,
                                             kernel_regularizer=keras.regularizers.L1(2.e-5))
                                             #bias_initializer=keras.initializers.HeNormal())
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="elu",
                            #kernel_regularizer=keras.regularizers.L1(1.0e-4),
                            bias_initializer=keras.initializers.HeNormal()
                          ),
             layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.conv1 = layers.Conv1D(filters=10, kernel_size=3, strides=1, padding="causal")#, input_shape=[None, embed_dim]),

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(inputs + attn_output)
        #ffn_output = self.conv1(out1)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()

        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.width = maxlen
        self.embed_dim = embed_dim

    def call(self, x):
        positions = tf.range(start=0, limit=self.width, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class FixedPositionEmbedding(layers.Layer):
    def __init__(self, seq_len, embed_dim, scaling=0.05):
        super(FixedPositionEmbedding, self).__init__()
        self.width = seq_len
        self.embed_dim = embed_dim
        self.scaling = scaling

    def call(self, x):
        positions = tf.map_fn(fn=lambda t: self.scaling / K.exp(t / 3), elems=tf.range(start=0, limit=self.width, delta=1, dtype=tf.float32))
        positions = tf.broadcast_to(tf.reshape(positions, [self.width,1]), [self.width, self.embed_dim])
        return x + positions
