import abc
from typing import List
import tensorflow as tf
import numpy as np


class AbstractAggregator(abc.ABC):
    def __init__(self) -> None:
        self.layers_delta: List[List[np.ndarray]] = []
        self.first_trainable_layer = 0

    def clear_aggregator(self):
        self.layers_delta = []

    def add_client_delta(self, client_delta: List[tf.Tensor]):
        if len(self.layers_delta) == 0:
            self.layers_delta = [[] for _ in range(len(client_delta))]
        for i, delta in enumerate(client_delta):
            self.layers_delta[i].append(np.array(delta))

    @abc.abstractmethod
    def aggregate(self, server_model: tf.keras.Model, num_byzantine: int):
        pass

    def set_first_trainable_layer(self, first_trainable_layer):
        self.first_trainable_layer = first_trainable_layer

    def apply_delta(self, server_model: tf.keras.Model, delta: List[np.ndarray]):
        old_weights = server_model.get_weights()
        updated_part = [old + d for old, d in zip(old_weights[self.first_trainable_layer:], delta)]
        new_weights = old_weights[:self.first_trainable_layer] + updated_part
        server_model.set_weights(new_weights)
        self.clear_aggregator()


class FedAvgAggregator(AbstractAggregator):
    def aggregate(self, server_model: tf.keras.Model, num_byzantine: int):
        delta = []
        for layer in self.layers_delta:
            delta.append(np.mean(layer, axis=0))
        self.apply_delta(server_model, delta)


class MedianAggregator(AbstractAggregator):
    def aggregate(self, server_model: tf.keras.Model, num_byzantine: int):
        delta = []
        for layer in self.layers_delta:
            delta.append(np.median(layer, axis=0))
        self.apply_delta(server_model, delta)


class TrimmedMeanAggregator(AbstractAggregator):

    def __init__(self, beta: float) -> None:
        super().__init__()
        if not (0 <= beta < 0.5):
            raise ValueError('beta shoulb be in in [0,1/2) interval')
        self.beta = beta

    def aggregate(self, server_model: tf.keras.Model, num_byzantine: int):
        delta = []
        num_clients = len(self.layers_delta[0])
        exclusions = int(np.round(2 * self.beta * num_clients))
        low = exclusions // 2
        high = exclusions // 2
        high += exclusions % 2
        high = num_clients - high
        if low == high:
            high = min(num_clients, high + 1)
        for i, layer in enumerate(self.layers_delta):
            layer = np.sort(layer, axis=0)[low:high]
            delta.append(np.mean(layer, axis=0))
        self.apply_delta(server_model, delta)


class MultiKrumAggregator(AbstractAggregator):

    def __init__(self, m) -> None:
        super().__init__()
        self.m = m

    def aggregate(self, server_model: tf.keras.Model, num_byzantine: int):
        num_clients = len(self.layers_delta[0])
        num_layers = len(server_model.get_weights())
        k = num_clients - num_byzantine - 2
        k = max(1, k)
        flattened_deltas = []
        for client_index in range(num_clients):
            client_data = []
            for layer_index in range(num_layers):
                client_data.append(np.array(self.layers_delta[layer_index][client_index]).flatten())
            flattened_deltas.append(np.concatenate(client_data))
        deltas = np.vstack(flattened_deltas)

        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                distances[i, j] = np.square(deltas[i] - deltas[j]).sum()
                distances[j, i] = distances[i, j]

        distances.sort(axis=0)
        best_clients = np.argsort(distances[:k + 1].sum(axis=0))[:min(self.m, num_clients)]
        delta = []
        for layer in self.layers_delta:
            delta.append(np.mean(np.stack(layer)[best_clients], axis=0))
        self.apply_delta(server_model, delta)


class KrumAggregator(MultiKrumAggregator):

    def __init__(self) -> None:
        super().__init__(1)
