from typing import Callable, Tuple

import os
import tensorflow as tf

from attacks.data_attacker import AbstractDataAttacker
from attacks.model_attacker import BackdoorAttack
from utils.constants import *
from fed.partitioner import get_partitioned_indices


class Dataset(object):

    def __init__(self, x_train, y_train, x_test, y_test,
                 data_preprocessor: Callable[[any, any], Tuple],
                 non_iid_deg: float = 0) -> None:
        self.x_train = x_train
        self.y_train = y_train.flatten()
        self.x_test = x_test
        self.y_test = y_test.flatten()
        self.partitioned_data = None
        self.example_input = None
        self.data_preprocessor = data_preprocessor
        self.partition_dataset(non_iid_deg)

    def partition_dataset(self, non_iid_deg: float):
        if self.x_train is None:
            return
        partitioned_indices = get_partitioned_indices(self.y_train, non_iid_deg)
        self.partitioned_data = [(self.x_train[client_indices],
                                  self.y_train[client_indices])
                                 for client_indices in partitioned_indices]

    def create_client_data(self, data):
        return tf.data.Dataset.from_tensor_slices(data) \
            .shuffle(SHUFFLE_BUFFER) \
            .batch(BATCH_SIZE) \
            .map(self.data_preprocessor) \
            .prefetch(PREFETCH_BUFFER)

    def make_federated_data(self):
        return [(client, self.create_client_data(self.partitioned_data[client])) for client in
                pick_clients(TRAINING_FRACTION)]

    def attack_data(self, data_attacker: AbstractDataAttacker):
        if data_attacker is not None:
            self.partitioned_data = data_attacker.attack(self.partitioned_data)

    def create_test_data(self):
        x, y = list(zip(*[self.data_preprocessor(x, y) for x, y in zip(self.x_test, self.y_test)]))
        return np.array(x), np.array(y)

    def backdoor(self, model_attacker: BackdoorAttack):
        self.partitioned_data = model_attacker.attack_train(self.partitioned_data)
        model_attacker.attack_test(self.x_test, self.y_test)


class ADNIDataset(Dataset):

    def __init__(self, root_dir: str):
        super(ADNIDataset, self).__init__(None, np.array([]), None, np.array([]), lambda x, y: (x, y))
        if not root_dir.endswith('/'):
            root_dir += '/'
        self.x_test = np.load(root_dir + 'x_test.npy')
        self.y_test = np.load(root_dir + 'y_test.npy')

        self.x_val = np.load(root_dir + 'x_val.npy')
        self.y_val = np.load(root_dir + 'y_val.npy')
        self.partitioned_data = []
        for folder in os.listdir(root_dir):
            path = root_dir + folder + '/'
            if os.path.isdir(path):
                self.partitioned_data.append((np.load(path + 'x_train.npy'), np.load(path + 'y_train.npy')))

    def create_client_data(self, data):
        gen_params = {"featurewise_center": False, "samplewise_center": False, "featurewise_std_normalization": False,
                      "samplewise_std_normalization": False, "zca_whitening": False, "rotation_range": 5,
                      "shear_range": 0.1, "horizontal_flip": True, "vertical_flip": True, "fill_mode": 'constant',
                      "cval": 0}
        gen = tf.keras.preprocessing.image.ImageDataGenerator(**gen_params)

        gen.fit(data[0], seed=1)

        return gen.flow(data[0], data[1], batch_size=BATCH_SIZE, shuffle=True)
