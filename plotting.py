import matplotlib.pyplot as plt
import numpy as np


def plot_series(time, series, subplot=None, format="-",  start=0, end=None, ):
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
    loss = history.history['loss']
    epochs = range(10, len(loss))
    plt.plot(epochs, loss[10:], 'b', label='Training Loss')
    plt.show()


def plot_fitting( model, data_series, pred, window_size):
    # forecast = []
    # for time in range(len(series) - window_size):
    #     input = series[time:time + window_size][np.newaxis]
    #     forecast.append(model.predict(input))

    no_plot = min( 10, pred.shape[1])

    #forecast = forecast[split_time - window_size:]
    #results = np.array(forecast)[:, 0, :]

    #plt.figure(figsize=(18, 6))
    figure, axis = plt.subplots(5, 2, figsize=(17,9))
    X = range(0, pred.shape[0])
    for n in range(0,no_plot):
        plot_series(X, data_series[window_size+1:-(window_size-1),n], axis[n//2, n%2])
        plot_series(X, pred[:,n], axis[n//2, n%2])

    plt.show()
