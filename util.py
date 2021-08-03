from typing import Callable, Tuple

import partitioner
from constants import *
from data_attacker import AbstractDataAttacker
import tensorflow as tf


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
        partitioned_indices = partitioner.get_partitioned_indices(self.y_train, non_iid_deg)
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
        self.partitioned_data = data_attacker.attack(self.partitioned_data)
