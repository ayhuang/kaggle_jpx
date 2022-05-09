import random

import matplotlib.pyplot as plt
import numpy as np


def plot_series(time, series, subplot=None, format="-",  start=0, end=None ):
    if subplot is None:
        plt.plot(time[start:end], series[start:end], format)
        #plt.xlabel("Time")
        #plt.ylabel("Value")
        plt.grid(True)
    else:
        subplot.plot(time[start:end], series[start:end], format)
        #subplot.xlabel("Time")
        #subplot.ylabel("Value")
        subplot.grid(True)


def plot_training_hist( history ):
    t_loss = history.history['loss']
    v_loss = history.history['val_loss']
    mae = history.history['mae']
    v_mae = history.history['val_mae']
    epochs = range(10, len(t_loss))
    plt.plot(epochs, t_loss[10:], 'b', label='Training Loss')
    plt.plot(epochs, v_loss[10:], 'k', label='Val Loss')
    plt.plot(epochs, mae[10:], 'g', label='mae')
    plt.plot(epochs, v_mae[10:], 'y', label='val_mae')

    plt.show()


def plot_fitting( model, data_series, pred, window_size):
    # forecast = []
    # for time in range(len(series) - window_size):
    #     input = series[time:time + window_size][np.newaxis]
    #     forecast.append(model.predict(input))

    no_plot = min( 4, pred.shape[1])

    #forecast = forecast[split_time - window_size:]
    #results = np.array(forecast)[:, 0, :]

    #plt.figure(figsize=(18, 6))
    figure, axis = plt.subplots(4, 1, figsize=(15,8))
    X = range(0, pred.shape[0])
    for i in range(0,no_plot):
        n = random.randint(0,1999)
        #plot_series(X, data_series[window_size+1:-(window_size-1),n], axis[i])#n%2])
        plot_series(X, data_series[window_size-1:, n,2], axis[i])  # n%2])
        plot_series(X, pred[:,n], axis[i])
    plt.show()
