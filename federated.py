import time
from typing import Callable, Tuple
import tensorflow as tf
import numpy as np

import partitioner
from aggregators import AbstractAggregator
from constants import *
from data_attacker import AbstractDataAttacker, NoDataAttacker
from model_attacker import AbstractModelAttacker, NoModelAttacker


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


class FedTester:
    def __init__(self, model_fn: Callable[[], tf.keras.Model],
                 dataset: Dataset,
                 aggregator: AbstractAggregator,
                 data_attacker: AbstractDataAttacker = NoDataAttacker(),
                 model_attacker: AbstractModelAttacker = None
                 ) -> None:
        self.model_fn = model_fn
        self.dataset = dataset
        self.aggregator = aggregator
        self.client_trainer = NoModelAttacker(self.model_fn,
                                              tf.keras.optimizers.SGD(),
                                              tf.keras.losses.SparseCategoricalCrossentropy())
        self.data_attacker = data_attacker
        self.model_attacker = model_attacker

    def initialize_federated(self):
        self.dataset.attack_data(self.data_attacker)
        server_model = self.model_fn()
        server_model.compile(metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        self.aggregator.clear_aggregator()
        test_data = self.dataset.create_client_data((self.dataset.x_test, self.dataset.y_test))
        return server_model, test_data

    def perform_fed_training(self, number_of_rounds: int = NUM_ROUNDS):
        t = time.time()
        server_model, test_data = self.initialize_federated()
        byzantine = 0
        if self.data_attacker is not None:
            byzantine = int(len(self.data_attacker.get_attacked_clients()) * TRAINING_FRACTION)
        for round_num in range(1, number_of_rounds + 1):
            for client, client_dataset in self.dataset.make_federated_data():
                self.aggregator.add_client_delta(self.client_trainer.forward_pass(client_dataset, server_model))
            self.aggregator.aggregate(server_model, byzantine)
            print('Training round: {}\t\taccuracy = {}'
                  .format(round_num, server_model.evaluate(test_data, verbose=0)[1]))

        print('Training duration {}'.format(time.time() - t))
