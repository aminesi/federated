import time
from typing import Callable
import tensorflow as tf

from fed.aggregators import AbstractAggregator
from utils.constants import *
from attacks.data_attacker import AbstractDataAttacker, NoDataAttacker
from attacks.model_attacker import AbstractModelAttacker, NoModelAttacker
from utils.util import Dataset


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
        self.benign_trainer = NoModelAttacker()
        self.data_attacker = data_attacker
        self.model_attacker = model_attacker
        self.first_trainable_layer = 0

    def initialize_federated(self):
        self.dataset.attack_data(self.data_attacker)
        server_model = self.model_fn()
        first_trainable_layer = 0
        for i, layer in enumerate(server_model.layers[::-1]):
            if not layer.trainable:
                first_trainable_layer = -i
                break
        self.set_first_trainable_layer(first_trainable_layer)
        server_model.compile(metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        self.aggregator.clear_aggregator()
        test_data = self.dataset.create_test_data()
        return server_model, test_data

    def perform_fed_training(self, number_of_rounds: int = NUM_ROUNDS):
        t = time.time()
        server_model, test_data = self.initialize_federated()
        byzantine = 0
        if self.data_attacker is not None:
            byzantine = int(len(self.data_attacker.get_attacked_clients()) * TRAINING_FRACTION)
        for round_num in range(1, number_of_rounds + 1):
            for client, client_dataset in self.dataset.make_federated_data():
                trainer = self.get_trainer(client)
                self.aggregator.add_client_delta(trainer.forward_pass(client_dataset, server_model))
            self.aggregator.aggregate(server_model, byzantine)
            print('Training round: {}\t\taccuracy = {}'
                  .format(round_num, server_model.evaluate(test_data, verbose=0)[1]))

        print('Training duration {}'.format(time.time() - t))

    def get_trainer(self, client):
        if self.model_attacker and client in self.model_attacker.get_attacked_clients():
            return self.model_attacker
        return self.benign_trainer

    def set_first_trainable_layer(self, first_trainable_layer):
        self.first_trainable_layer = first_trainable_layer
        self.benign_trainer.set_first_trainable_layer(first_trainable_layer)
        self.aggregator.set_first_trainable_layer(first_trainable_layer)
