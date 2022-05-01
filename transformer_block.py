import tensorflow as tf
from tensorflow import keras
from keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,
                                             kernel_initializer="glorot_uniform",
                                             dropout=rate,
                                             kernel_regularizer=keras.regularizers.L1(2.e-4),#L2(l1=0.001, l2=0.001),
                                             bias_initializer=keras.initializers.HeNormal())

        # self.att_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,
        #                                      kernel_initializer="glorot_uniform",
        #                                     kernel_regularizer=keras.regularizers.L1L2( l1=0.2,l2=0.3),
        #                                   #   bias_regularizer=keras.regularizers.L2( 0.2 ),
        #                                      bias_initializer=keras.initializers.HeNormal())
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="elu",

                               #    kernel_regularizer=keras.regularizers.L1L2(l1=0.05, l2=0.05),
                               # bias_regularizer=keras.regularizers.L2(0.01),
                            bias_initializer=keras.initializers.HeNormal()
                          ),
             layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
      #  self.conv1 = layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="causal")# input_shape=[None, embed_dim]),

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(inputs + attn_output)
       # out1 = self.conv1(out1)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

#Two seperate embedding layers, one for tokens, one for token index (positions).

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
  #      self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.width = maxlen

    def call(self, x):
        #maxlen = tf.shape(x)[0].shape
        positions = tf.range(start=0, limit=self.width, delta=1)
        positions = self.pos_emb(positions)
        return x + positions