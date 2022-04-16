import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

dataset = tf.data.Dataset.range(10)
for i in dataset:
    print(i.numpy())
dataset = dataset.window(5, shift=1, drop_remainder=True)
for i in dataset:
   for val in i:
       print(val.numpy(), end=' ')
   print()

dataset = dataset.flat_map(lambda window: window.batch(5))
for i in dataset:
   print(i)
   print( i.numpy() )


dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)

for(x, y) in dataset:
    print("x= ", x.numpy(), "y= ", y.numpy())
