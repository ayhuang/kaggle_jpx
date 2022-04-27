import random

import numpy as np
import pandas as pd
#import jpx_tokyo_market_prediction
import plotting
import sys

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import transformer_block
print(tf.__version__)
import warnings
warnings.filterwarnings("ignore")


embed_dim = 2000  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 16  # Hidden layer size in feed forward network inside transformer
window_size = 15
batch_size = 64
no_epoches = 15000

train_ds = tf.data.experimental.load( "train_files/train_dataset")
val_ds = tf.data.experimental.load( "train_files/val_dataset")

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    inputs= layers.Input(shape=( window_size, embed_dim))
    x= layers.BatchNormalizationV2()(inputs)
    embedding_layer = transformer_block.TokenAndPositionEmbedding(window_size, 0, embed_dim)
    x = embedding_layer(x)
    tb_1 = transformer_block.TransformerBlock(embed_dim, num_heads, ff_dim, rate=0.4)
 #   tb_2 = transformer_block.TransformerBlock(embed_dim, num_heads, 10, rate=0.1)

    x = tb_1(x)
 #   x = tb_2(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(8, activation='elu')(x)
    #x = layers.Dense(10, activation='relu')(x)
    x = layers.Dense(embed_dim, activation='elu')(x)
    outputs = layers.Dense(embed_dim)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler( lambda epoch: 1.e-1 if epoch >2000 else 2.0e-1)# * 10**(-int(epoch / 1000)))
    #opt = tf.keras.optimizers.SGD(learning_rate=5.0e-1, momentum=0.8)
    opt = tf.keras.optimizers.Adam(learning_rate=5.0e-1, epsilon=1)
    model.compile( optimizer=opt, loss=keras.losses.Huber(), metrics=["mae"])

    model.summary()
    history = model.fit( train_ds, epochs=no_epoches, validation_data=val_ds)# callbacks=[lr_schedule])

#history = model.fit( x_train,y_train, batch_size=32, epochs=100) #, validation_data=val_ds)# callbacks=[lr_schedule])
#
plotting.plot_training_hist( history )

train_pred = model.predict( train_ds, batch_size = batch_size)
val_pred = model.predict( val_ds, batch_size = batch_size )

#plotting.plot_fitting( model, daily_price_series, np.concatenate((train_pred, val_pred), axis=0), window_size)




