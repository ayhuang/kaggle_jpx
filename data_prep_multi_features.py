import random

import numpy as np
import pandas as pd
#import jpx_tokyo_market_prediction
import plotting
import sys
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

stocks_to_train = []#[ all_stock_codes[i] for i in random.sample(range(0,len( all_stock_codes)),1000)]
#print( f'stock selected: {stocks_to_train}')
#stocks_to_train = [2264, 1332, 1333, 9990, 9991, 9997]
if len(stocks_to_train) != 0:
    prices = prices[prices.SecuritiesCode.isin(stocks_to_train)]


enc = OrdinalEncoder(dtype=np.int)
prices["SecuritiesCode"] = enc.fit_transform(prices[["SecuritiesCode"]])


codes = list(prices.SecuritiesCode.unique())
date_list = list(prices.Date.unique())
codes_size = len(codes)

prices = prices[['Date', 'SecuritiesCode', 'Close', 'Volume', 'Target']].dropna()
#normalize the data
prices["Close"] = prices["Close"]/100000
prices["Volume"] = prices["Volume"]/6.0e7
prices["Target"] = prices["Target"]*100

# not all days have the same number of stocks, so we need to pad the missing data
def pad_missing_stock_code( sample, codes):
    # missing code
    missing_codes = set( list( range(0, len(codes)))) - set( [i[0] for i in sample])
    # drop the code column
    x = sample[:,1:]
    for idx in sorted(missing_codes):
        x = np.insert( x, idx, 0.0, axis=0)
    return x


def windowed_dataset(series, window_size, batch_size):
    #series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
  #  ds = ds.shuffle( len( series))
    # first 2 columns are closing price, volume, last column is the label - target
    ds = ds.map(lambda w: (w[:,:,0:2], w[-1,:,2].reshape(-1)))
    #if batch_size == 1: return ds
    return ds.batch(batch_size).prefetch(1)



# prep time series data set for training and validation
def prep_dataset( prices,  window_size, batch_size):
    price_series = prices.sort_values(by=['Date', 'SecuritiesCode']).reset_index(drop=True).dropna()

    daily_data_list =[]
    for dt in date_list:
        daily_data = price_series[price_series.Date == dt ].drop(['Date'], axis=1).sort_values(by=['SecuritiesCode'])
        daily_data_list.append( pad_missing_stock_code( daily_data.to_numpy(), codes))

    # daily_target_list is a 1201 long list of 1-d (2000) array
    # split = int(len(daily_target_list)*0.90)
    # train = daily_target_list[:split]
    # val = daily_target_list[split:]

    ds = windowed_dataset( daily_data_list, window_size, batch_size )
    #ds_val = windowed_dataset( val, window_size, batch_size
    return ds, np.array(daily_data_list)

window_size = 7
batch_size = 64
ds, daily_price_series = prep_dataset( prices, window_size, batch_size)

tf.data.experimental.save( ds, "train_files/price_dataset")
#tf.data.experimental.save( val_ds, "train_files/val_dataset")
np.save( "train_files/daily_series.npy", daily_price_series)
sys.exit()




