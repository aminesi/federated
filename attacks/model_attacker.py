import abc

import numpy as np
import tensorflow as tf

from attacks.attacker import AbstractAttacker
from config import get_model, get_optimizer, get_loss
from utils.constants import NUM_EPOCHS, pick_clients


class AbstractModelAttacker(AbstractAttacker):
    @abc.abstractmethod
    def forward_pass(self, dataset: tf.data.Dataset, server_model: tf.keras.Model):
        pass


class NoModelAttacker(AbstractModelAttacker):

    def __init__(self) -> None:
        super().__init__(0)
        self.model_fn = get_model
        self.optimizer = get_optimizer()
        self.loss = get_loss()

    def forward_pass(self, dataset: tf.data.Dataset, server_model: tf.keras.Model):
        local_model = self.model_fn()
        old_weights = server_model.get_weights()
        local_model.set_weights(old_weights)
        local_model.compile(
            optimizer=self.optimizer,
            loss=self.loss
        )
        local_model.fit(dataset, epochs=NUM_EPOCHS, verbose=0)
        weights_delta = tf.nest.map_structure(tf.subtract, local_model.get_weights(), old_weights)
        return weights_delta


class RandomModelAttacker(AbstractModelAttacker):

    def __init__(self, fraction: float, std: float) -> None:
        super().__init__(fraction)
        self.std = std
        self.attacked_clients = set(pick_clients(fraction))

    def forward_pass(self, dataset: tf.data.Dataset, server_model: tf.keras.Model):
        return [np.random.normal(0, self.std, layer_weight.shape) for layer_weight in server_model.get_weights()]


class SignFlipModelAttacker(NoModelAttacker):

    def __init__(self, fraction: float, multiplier: float) -> None:
        super().__init__()
        self.fraction = fraction
        self.multiplier = multiplier
        self.attacked_clients = set(pick_clients(fraction))

    def forward_pass(self, dataset: tf.data.Dataset, server_model: tf.keras.Model):
        clean_grads = super(SignFlipModelAttacker, self).forward_pass(dataset, server_model)
        return [layer * -1 * self.multiplier for layer in clean_grads]
