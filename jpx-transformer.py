import random

import numpy as np
import pandas as pd
#import jpx_tokyo_market_prediction
import plotting

from sklearn.preprocessing import OrdinalEncoder

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import transformer_block
print(tf.__version__)
import warnings
warnings.filterwarnings("ignore")

prices = pd.read_csv("train_files/stock_prices.csv")
pd.options.display.width = None

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
prices.head(5)
all_stock_codes = list(prices.SecuritiesCode.unique())

stocks_to_train = [ all_stock_codes[i] for i in random.sample(range(0,len( all_stock_codes)),500)]
#print( f'stock selected: {stocks_to_train}')
#stocks_to_train = [2264, 1332, 1333, 9990, 9991, 9997]
if len(stocks_to_train) != 0:
    prices = prices[prices.SecuritiesCode.isin(stocks_to_train)]


enc = OrdinalEncoder(dtype=np.int)
prices["SecuritiesCode"] = enc.fit_transform(prices[["SecuritiesCode"]])
#normalize the change to percentage
prices["Target"] = prices["Target"]*100

codes = list(prices.SecuritiesCode.unique())
date_list = list(prices.Date.unique())
codes_size = len(codes)

prices = prices[['Date', 'SecuritiesCode', 'Target']].dropna()


def pad_missing_target( sample, codes):
    # missing code
    missing_codes = set( list( range(0, len(codes)))) - set( [i[0] for i in sample])
    x = sample[:,1]
    for idx in missing_codes:
        x = np.insert( x, idx, 0.0)

    return x


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    #series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:].reshape(-1)))
    #if batch_size == 1: return ds
    return ds.batch(batch_size).prefetch(1)


# prep time series data set for training and validation
def prep_dataset( prices,  window_size, batch_size):
    price_series = prices.sort_values(by=['Date', 'SecuritiesCode']).reset_index(drop=True).dropna()

    daily_target_list =[]
    for dt in date_list:
        one_day_target = price_series[price_series.Date == dt ].drop(['Date'], axis=1).sort_values(by=['SecuritiesCode'])
        daily_target_list.append( pad_missing_target( one_day_target.to_numpy(), codes))

    # daily_target_list is a 1201 long list of 1-d (2000) array
    split = int(len(daily_target_list)*0.80)
    train = daily_target_list[:split]
    val = daily_target_list[split:]

    ds_train = windowed_dataset( train, window_size, batch_size, len(train) )
    ds_val = windowed_dataset( val, window_size, 1, len(train))
    x_train = []
    y_train = []
    for x,y in ds_train:
        x_train.append( x.numpy())
        y_train.append( y.numpy())


   # return np.array(x_train), np.array(y_train).reshape(-1), np.array(daily_target_list)
    return ds_train,ds_val, np.array(daily_target_list)


embed_dim = codes_size  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 20  # Hidden layer size in feed forward network inside transformer
window_size = 20
batch_size = 64
#(x_train, y_train, daily_price_series) = prep_dataset( prices, window_size, batch_size)
#print(f"x_train shape {x_train.shape}, y_train shape {y_train.shape}")
train_ds,val_ds, daily_price_series = prep_dataset( prices, window_size, batch_size)


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    inputs= layers.Input(shape=( window_size,embed_dim))
    embedding_layer = transformer_block.TokenAndPositionEmbedding(window_size, 0, embed_dim)
    x = embedding_layer(inputs)
    tb_1 = transformer_block.TransformerBlock(embed_dim, num_heads, ff_dim, rate=0.0)
    tb_2 = transformer_block.TransformerBlock(embed_dim, num_heads, 10, rate=0.1)

    x = tb_1(x)
    x = tb_2(x)
    x = layers.GlobalAveragePooling1D()(x)
    #x = layers.Dropout(0.2)(x)
    x = layers.Dense(20, activation='elu')(x)
    #x = layers.Dense(10, activation='relu')(x)
    x = layers.Dense(embed_dim, activation='elu')(x)
    outputs = layers.Dense(embed_dim)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler( lambda epoch: 1e-4 if epoch >1001 else 8e-3 * 10**(-int(epoch / 500)))
    #opt = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=1.0e-3)#, momentum=0.9)
    model.compile( optimizer=opt, loss="mse", metrics = ["mae"])#tf.keras.losses.Huber(), metrics=["mse"])

    model.summary()
    history = model.fit( train_ds, epochs=20, validation_data=val_ds)#, callbacks=[lr_schedule])

#history = model.fit( x_train,y_train, batch_size=32, epochs=100) #, validation_data=val_ds)# callbacks=[lr_schedule])
#
plotting.plot_training_hist( history )

train_pred = model.predict( train_ds, batch_size = batch_size)
val_pred = model.predict( val_ds, batch_size = batch_size )

plotting.plot_fitting( model, daily_price_series, np.concatenate((train_pred, val_pred), axis=0), window_size)
#
#
# env = jpx_tokyo_market_prediction.make_env()
# iter_test = env.iter_test()
#
# for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
#     prices["SecuritiesCode"] = enc.fit_transform(prices[["SecuritiesCode"]])
#
#     X_test = prices[["SecuritiesCode", "Open", "High", "Low", "Close"]]
#     lgbm_preds = list()
#     for model in lgbm_models:
#         lgbm_preds.append( model.predict(X_test) )
#     lgbm_preds = np.mean(lgbm_preds, axis=0)
#
#     X_test_prices = prices[["Open", "High", "Low", "Close"]]
#     X_test_id = prices[["SecuritiesCode"]]
#     dnn_preds = list()
#     for model in dnn_models:
#         dnn_preds.append( model.predict([X_test_prices, X_test_id]) )
#     dnn_preds = np.mean(lgbm_preds, axis=0)
#
#     sample_prediction["Prediction"] = lgbm_preds*0.5 + dnn_preds*0.5
#
#     sample_prediction = sample_prediction.sort_values(by = "Prediction", ascending=False)
#     sample_prediction.Rank = np.arange(0,2000)
#     sample_prediction = sample_prediction.sort_values(by = "SecuritiesCode", ascending=True)
#     sample_prediction.drop(["Prediction"],axis=1)
#     submission = sample_prediction[["Date","SecuritiesCode","Rank"]]
#     env.predict(submission)
#

# In[ ]:




