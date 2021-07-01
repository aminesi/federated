import time
from typing import Callable, OrderedDict
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

import partitioner
from attack import BaseAttacker

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
    def __init__(self, model: Callable[[], tf.keras.Model], dataset: Dataset, preprocessor: Callable[[any, any], OrderedDict], attacker: BaseAttacker) -> None:
        self.keras_model = model
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.attacker = attacker
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

    def create_client_data(self, data):
        return tf.data.Dataset.from_tensor_slices(data) \
            .repeat(NUM_EPOCHS) \
            .shuffle(SHUFFLE_BUFFER) \
            .batch(BATCH_SIZE) \
            .map(self.preprocessor) \
            .prefetch(PREFETCH_BUFFER)

    def make_federated_data(self):
        return [self.create_client_data(self.partitioned_data[client]) for client in FedTester.pick_clients(TRAINING_FRACTION)]

    def model_fn(self):
        # We _must_ create a new model here, and _not_ capture it from an external
        # scope. TFF will call this within different graph contexts.
        keras_model = self.keras_model()
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=self.example_input,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def perform_fed_training(self, number_of_rounds: int = NUM_ROUNDS):
        self.partitioned_data = self.attacker.attack(self.partitioned_data)
        self.example_input = self.create_client_data(self.partitioned_data[0]).element_spec
        test_data = [self.create_client_data((tf.convert_to_tensor(self.dataset.x_test),
                                              tf.convert_to_tensor(self.dataset.y_test)))]

        iterative_process = tff.learning.build_federated_averaging_process(
            self.model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

        state = iterative_process.initialize()
        evaluation = tff.learning.build_federated_evaluation(self.model_fn)

        t = time.time()
        for round_num in range(1, number_of_rounds + 1):
            state, metrics = iterative_process.next(state, self.make_federated_data())
            # print('round {:2d}, metrics={}'.format(round_num, metrics))
            print('round {:2d}, test results={}'.format(round_num, evaluation(state.model, test_data)))

        print(time.time() - t)
