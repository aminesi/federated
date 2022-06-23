import json
from typing import Dict
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
import os
import shutil
import copy

ADNI_ROOT = os.environ.get('ADNI_ROOT', './adni')
RESULTS_ROOT = os.environ.get('RESULTS_ROOT', './results')
CONFIG_PATH = os.environ.get('CONFIG_PATH', './config.json')

with open(CONFIG_PATH) as config_file:
    config: Dict[str, any] = json.load(config_file)
    config_file.close()

possible_configs = {
    'dataset': {'adni', 'mnist', 'cifar'},
    'aggregator': {'fed-avg', 'median', 'trimmed-mean', 'krum', 'combine'},
    'attack': {'label-flip', 'noise-data', 'overlap-data', 'delete-data', 'unbalance-data', 'random-update',
               'sign-flip', 'backdoor'},
    'attack-fraction': 'float between 0 and 1',
    'non-iid-deg': 'float between 0 and 1',
    'num-rounds': 'integer value'

}

possible_attacks = {'label-flip', 'noise-data', 'overlap-data', 'delete-data', 'unbalance-data'}


def throw_conf_error(param):
    raise AttributeError('\n\nConfig file does not have proper value for "{}"\nAcceptable values are: {}'
                         .format(param, possible_configs[param]))


def get_param(param, default_val=None):
    if param in config:
        return config[param]
    if default_val is not None:
        return default_val
    throw_conf_error(param)


def get_non_iid_deg():
    return get_param('non-iid-deg')


def load_data():
    x_train, y_train, x_test, y_test = None, None, None, None
    if config['dataset'] == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        y_test = copy.deepcopy(y_test)
        x_test = copy.deepcopy(x_test)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    elif config['dataset'] == 'cifar':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif config['dataset'] == 'adni':
        return ADNI_ROOT
    else:
        throw_conf_error('dataset')
    return x_train, y_train, x_test, y_test, lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), get_non_iid_deg()


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
    elif config['dataset'] == 'adni':
        model = VGG16(include_top=False, input_shape=(256, 256, 3))
        model.get_layer('block1_conv1').trainable = False
        model.get_layer('block1_conv2').trainable = False
        model.get_layer('block2_conv1').trainable = False
        model.get_layer('block2_conv2').trainable = False
        model.get_layer('block3_conv1').trainable = False
        model.get_layer('block3_conv2').trainable = False
        model.get_layer('block3_conv3').trainable = False
        model.get_layer('block4_conv1').trainable = False
        model.get_layer('block4_conv2').trainable = False
        model.get_layer('block4_conv3').trainable = False
        model.get_layer('block5_conv1').trainable = False
        model.get_layer('block5_conv2').trainable = False
        model.get_layer('block5_conv3').trainable = False

        # add new classifier layers
        flat1 = tf.keras.layers.Flatten()(model.layers[-1].output)
        class1 = tf.keras.layers.Dense(512, activation='relu')(flat1)
        output = tf.keras.layers.Dense(2, activation='softmax')(class1)
        # define new model
        model = tf.keras.Model(inputs=model.inputs, outputs=output)

        return model
    else:
        throw_conf_error('dataset')


def get_optimizer():
    if config['dataset'] == 'mnist':
        return tf.keras.optimizers.SGD(0.1)
    elif config['dataset'] == 'cifar':
        return tf.keras.optimizers.SGD(0.01)
    elif config['dataset'] == 'adni':
        return tf.keras.optimizers.SGD(0.001)
    else:
        throw_conf_error('dataset')


def get_loss():
    return tf.keras.losses.SparseCategoricalCrossentropy()


def get_num_round():
    return get_param('num-rounds')


def get_result_dir():
    attack = get_param('attack', 'clean')
    if attack != 'clean':
        attack = '{}-{}'.format(attack.replace('-', ''), get_param('attack-fraction'))
        if get_param('attack', 'clean') == 'noise-data':
            attack += '-sigma_multiplier-' + str(get_param('sigma_multiplier', 1))
        elif get_param('attack', 'clean') == 'overlap-data':
            attack += '-overlap_percentage-' + str(get_param('overlap_percentage', 0.75))
        elif get_param('attack', 'clean') == 'delete-data':
            attack += '-delete_percentage-' + str(get_param('delete_percentage', 0.75))
        elif get_param('attack', 'clean') == 'unbalance-data':
            attack += '-unbalance_percentage-' + str(get_param('unbalance_percentage', 0.75))
    dir_name = '{}-{}--{}--{}/'.format(get_param('dataset'), get_non_iid_deg(),
                                       get_param('aggregator', 'fed-avg').replace('-', ''),
                                       attack)
    path = os.path.join(RESULTS_ROOT, dir_name)
    os.makedirs(path, exist_ok=True)
    shutil.copy(CONFIG_PATH, path)
    return path
