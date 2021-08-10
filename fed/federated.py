import time
from typing import Callable, Dict, Any
import tensorflow as tf

from fed.aggregators import AbstractAggregator
from utils.constants import *
from attacks.data_attacker import AbstractDataAttacker, NoDataAttacker
from attacks.model_attacker import AbstractModelAttacker, NoModelAttacker, BackdoorAttack
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
        if self.model_attacker is not None and isinstance(self.model_attacker, BackdoorAttack):
            self.dataset.backdoor(self.model_attacker)
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

    def perform_fed_training(self, number_of_rounds: int = NUM_ROUNDS) -> Dict[str, Any]:
        results = {}
        t = time.time()
        server_model, test_data = self.initialize_federated()
        attacker = self.data_attacker
        if self.model_attacker:
            attacker = self.model_attacker
        byzantine = int(len(attacker.get_attacked_clients()) * TRAINING_FRACTION)
        for round_num in range(1, number_of_rounds + 1):
            data = self.dataset.make_federated_data()
            if self.model_attacker and isinstance(self.model_attacker, BackdoorAttack):
                attackers = 0
                for client, _ in data:
                    if client in self.model_attacker.get_attacked_clients():
                        attackers += 1
                self.model_attacker.chosen_attackers = attackers
            for client, client_dataset in data:
                trainer = self.get_trainer(client)
                self.aggregator.add_client_delta(trainer.forward_pass(client_dataset, server_model))
            self.aggregator.aggregate(server_model, byzantine)
            main_accuracy = server_model.evaluate(test_data, verbose=0)[1]
            if 'main_accuracy' not in results:
                results['main_accuracy'] = []
            results['main_accuracy'].append(main_accuracy)
            print('Training round: {}\t\taccuracy = {}'.format(round_num, main_accuracy))
            if self.model_attacker is not None and isinstance(self.model_attacker, BackdoorAttack):
                back_data = self.dataset.create_test_data(self.model_attacker.x_test, self.model_attacker.y_test)
                backdoor_accuracy = server_model.evaluate(back_data, verbose=0)[1]
                if 'backdoor_accuracy' not in results:
                    results['backdoor_accuracy'] = []
                results['backdoor_accuracy'].append(backdoor_accuracy)
                print('backdoor accuracy: {}'.format(backdoor_accuracy))

        t = time.time() - t
        results['time'] = t
        print('Training duration {}'.format(t))

        return results

    def get_trainer(self, client):
        if self.model_attacker and client in self.model_attacker.get_attacked_clients():
            return self.model_attacker
        return self.benign_trainer

    def set_first_trainable_layer(self, first_trainable_layer):
        self.first_trainable_layer = first_trainable_layer
        self.benign_trainer.set_first_trainable_layer(first_trainable_layer)
        if self.model_attacker:
            self.model_attacker.set_first_trainable_layer(first_trainable_layer)
        self.aggregator.set_first_trainable_layer(first_trainable_layer)
