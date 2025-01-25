import numpy as np
import tensorflow as tf
from keras.api.backend import sigmoid

def swish(x, beta = .1):
    return (x * sigmoid(beta * x))

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))