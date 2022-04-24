

import numpy as np
import pandas as pd
#import jpx_tokyo_market_prediction
import plotting

from sklearn.preprocessing import OrdinalEncoder

import tensorflow as tf
from tensorflow import keras
from keras import layers

import transformer_block
print(tf.__version__)

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

prices = pd.read_csv("train_files/stock_prices.csv")
pd.options.display.width = None

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
prices.head(5)

stock_list = [2264]
if len(stock_list) != 0:
    prices = prices[prices.SecuritiesCode.isin( stock_list )]


enc = OrdinalEncoder(dtype=int)
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
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
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
    target_array = np.array(daily_target_list)
    target_array = target_array.reshape( target_array.shape[0])
    train = target_array[:split]
    val = target_array[split:]

    ds_train = windowed_dataset( train, window_size, batch_size, len(train) )
    ds_val = windowed_dataset( val, window_size, 1, len(train))

   # return np.array(x_train), np.array(y_train).reshape(-1), np.array(daily_target_list)
    return ds_train,ds_val, np.array(daily_target_list)

def test_prep_dateset():
    (train_ds, val_ds, time_series) = prep_dataset( prices, 20, 32 )
   # print(len( val_ds ))
    for windows in train_ds.take(1):
        print(f'data type: {type(windows)}')
        print(f'number of elements in the tuple: {len(windows)}')
        print(f'shape of first element: {windows[0].shape}')

        print(f'shape of second element: {windows[1].shape}')
        print()
#test_prep_dateset()


window_size = 20
batch_size = 32
#(x_train, y_train, daily_price_series) = prep_dataset( prices, window_size, batch_size)
#print(f"x_train shape {x_train.shape}, y_train shape {y_train.shape}")
train_ds,val_ds, daily_price_series = prep_dataset( prices, window_size, batch_size)
for windows in train_ds.take(1):
  print(f'data type: {type(windows)}')
  print(f'number of elements in the tuple: {len(windows)}')
  print(f'shape of first element: {windows[0].shape}')
  print(f'shape of second element: {windows[1].shape}')

inputs= layers.Input(shape=( window_size))
outputs = layers.Dense(1)(inputs)

model = keras.Model(inputs=inputs, outputs=outputs)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
#     tf.keras.layers.Dense(10, activation="relu"),
#     tf.keras.layers.Dense(1, activation='linear')
#])

#lr_schedule = keras.callbacks.LearningRateScheduler( lambda epoch: 1e-4 * 10**(int(epoch / 20)))
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)# momentum=0.9)
model.compile( optimizer=opt, loss="mae", metrics=["mse"])

model.summary()
history = model.fit( train_ds, epochs=100, validation_data=val_ds)# callbacks=[lr_schedule])
print( model.get_layer(name="dense").get_weights())

plotting.plot_fitting( model, daily_price_series, window_size, 0)
