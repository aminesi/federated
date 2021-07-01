import collections
import os

from attack import NoAttacker, LabelFlipping

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from util import FedTester, Dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

dataset = Dataset(x_train, y_train, x_test, y_test)


def keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        # tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        # tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        # tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])


fed_tester = FedTester(
    keras_model,
    dataset,
    lambda x, y: collections.OrderedDict(
        x=tf.reshape(x, (-1, 784)),
        y=tf.reshape(y, (-1, 1)),
    ),
    LabelFlipping(0)
)

fed_tester.perform_fed_training(30)
