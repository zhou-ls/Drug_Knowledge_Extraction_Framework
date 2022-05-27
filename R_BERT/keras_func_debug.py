# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Add, Lambda, Multiply, Average, average, Flatten
from tensorflow.keras.models import Model
import numpy as np


if __name__ == '__main__':

    batch_size = 2

    a = K.constant([[0, 1, 1, 0],
                    [1, 0, 0, 1]])
    b = np.random.random(size=(batch_size, 4, 3))
    print(b)
    b = K.constant(b)
    c = K.batch_dot(a, b, axes=1)

    with tf.Session() as sess:
        output_array = sess.run(c)
        print(output_array)
        print(a.shape, b.shape, output_array.shape)

