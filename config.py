import json
from typing import Dict
import tensorflow as tf
import numpy as np

from util import Dataset

with open('config.json') as config_file:
    config: Dict[str, any] = json.load(config_file)
    config_file.close()

possible_configs = {
    'dataset': {'mnist', 'cifar'},
    'aggregator': {'fed-avg', 'median', 'tr-mean', 'krum', 'm-krum'},
    'attack': {'type', 'fraction', 'args'},
    'non_iid_deg': 'float between 0 and 1'

}

possible_attacks = {'label-flip', 'noise-data', 'overlap-data', 'delete-data', 'unbalance-data'}


def throw_conf_error(param):
    raise AttributeError('\n\nConfig file does not have proper value for "{}"\nAcceptable values are: {}'
                         .format(param, possible_configs[param]))


def get_non_iid_deg():
    if 'non_iid_deg' in config:
        return config['non_iid_deg']
    throw_conf_error('non_iid_deg')


def load_data():
    x_train, y_train, x_test, y_test = None, None, None, None
    if config['dataset'] == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    elif config['dataset'] == 'cifar':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        throw_conf_error('dataset')
    return Dataset(x_train, y_train, x_test, y_test, lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                   get_non_iid_deg())


def get_model():
    if config['dataset'] == 'mnist':
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
    elif config['dataset'] == 'cifar':
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=(32, 32, 3), activation='relu'),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])
    else:
        throw_conf_error('dataset')


def get_optimizer():
    if config['dataset'] == 'mnist':
        return tf.keras.optimizers.SGD(0.1)
    elif config['dataset'] == 'cifar':
        return tf.keras.optimizers.SGD(0.01)
    else:
        throw_conf_error('dataset')


def get_loss():
    if config['dataset'] == 'mnist':
        return tf.keras.losses.SparseCategoricalCrossentropy()
    elif config['dataset'] == 'cifar':
        return tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        throw_conf_error('dataset')
