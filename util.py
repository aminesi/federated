import time
from typing import Callable, Tuple
import tensorflow as tf
import numpy as np

import partitioner
from aggregators import AbstractAggregator
from data_attacker import AbstractDataAttacker, NoDataAttacker
from model_attacker import AbstractModelAttacker, NoModelAttacker

NUM_CLIENTS = 100
TRAINING_FRACTION = .1
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
NUM_ROUNDS = 100


class Dataset(object):

    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train.flatten()
        self.x_test = x_test
        self.y_test = y_test.flatten()


class FedTester:
    def __init__(self, model_fn: Callable[[], tf.keras.Model],
                 dataset: Dataset,
                 data_preprocessor: Callable[[any, any], Tuple],
                 aggregator: AbstractAggregator,
                 data_attacker: AbstractDataAttacker = NoDataAttacker(),
                 model_attacker: AbstractModelAttacker = None
                 ) -> None:
        self.model_fn = model_fn
        self.dataset = dataset
        self.data_preprocessor = data_preprocessor
        self.aggregator = aggregator
        self.client_trainer = NoModelAttacker(self.model_fn,
                                              tf.keras.optimizers.SGD(),
                                              tf.keras.losses.SparseCategoricalCrossentropy())
        self.data_attacker = data_attacker
        self.model_attacker = model_attacker
        self.partitioned_data = None
        self.example_input = None
        self.partition_dataset()

    def partition_dataset(self, non_iid_deg: float = 0):
        partitioned_indices = partitioner.get_partitioned_indices(self.dataset.y_train, non_iid_deg)
        self.partitioned_data = [(tf.convert_to_tensor(self.dataset.x_train[client_indices]),
                                  tf.convert_to_tensor(self.dataset.y_train[client_indices]))
                                 for client_indices in partitioned_indices]

    @staticmethod
    def pick_clients(fraction: float):
        count = int(fraction * NUM_CLIENTS)
        return np.random.choice(range(NUM_CLIENTS), replace=False, size=count)

    def create_client_data(self, data, epochs=NUM_EPOCHS):
        return tf.data.Dataset.from_tensor_slices(data) \
            .repeat(epochs) \
            .shuffle(SHUFFLE_BUFFER) \
            .batch(BATCH_SIZE) \
            .map(self.data_preprocessor) \
            .prefetch(PREFETCH_BUFFER)

    def make_federated_data(self):
        return [(client, self.create_client_data(self.partitioned_data[client])) for client in
                FedTester.pick_clients(TRAINING_FRACTION)]

    def initialize_federated(self):
        self.partitioned_data = self.data_attacker.attack(self.partitioned_data)
        server_model = self.model_fn()
        server_model.compile(metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        self.aggregator.clear_aggregator()
        test_data = self.create_client_data((self.dataset.x_test, self.dataset.y_test), 1)
        return server_model, test_data

    def perform_fed_training(self, number_of_rounds: int = NUM_ROUNDS):
        t = time.time()
        server_model, test_data = self.initialize_federated()
        byzantine = 0
        if self.data_attacker is not None:
            byzantine = int(len(self.data_attacker.get_attacked_clients())*TRAINING_FRACTION)
        for round_num in range(1, number_of_rounds + 1):
            for client, client_dataset in self.make_federated_data():
                self.aggregator.add_client_delta(self.client_trainer.forward_pass(client_dataset, server_model))
            self.aggregator.aggregate(server_model, byzantine)
            print('Training round: {}\t\taccuracy = {}'
                  .format(round_num, server_model.evaluate(test_data, verbose=0)[1]))

        print('Training duration {}'.format(time.time() - t))
