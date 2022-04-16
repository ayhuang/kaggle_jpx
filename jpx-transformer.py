

import numpy as np
import pandas as pd
#import jpx_tokyo_market_prediction

from sklearn.preprocessing import OrdinalEncoder

import tensorflow as tf
from tensorflow import keras
from keras import layers

import transformer_block
print(tf.__version__)
import warnings
warnings.filterwarnings("ignore")

prices = pd.read_csv("train_files/stock_prices.csv")
pd.options.display.width = None

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
prices.head(5)

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
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
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
    ds_val = windowed_dataset( val, window_size, batch_size, len(train))
    return (ds_train, ds_val )



def test_prep_dateset():
    (train_ds, val_ds) = prep_dataset( prices, 20, 32 )
    print( train_ds )

#test_prep_dateset()


embed_dim = codes_size  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 5  # Hidden layer size in feed forward network inside transformer
window_size = 20
batch_size = 32
(train_ds, val_ds) = prep_dataset( prices, window_size, batch_size)

inputs = layers.Input(shape=(window_size, embed_dim))
embedding_layer = transformer_block.TokenAndPositionEmbedding(window_size, 0, embed_dim)
x = embedding_layer(inputs)
transformer_block = transformer_block.TransformerBlock(embed_dim, num_heads, ff_dim, rate=0.3)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(2000, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(embed_dim)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile( optimizer="adam", loss=tf.keras.losses.Huber(), metrics=["mae"])

model.summary()


history = model.fit( train_ds,  epochs=200)


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




