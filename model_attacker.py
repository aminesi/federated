import abc
from typing import Callable

import tensorflow as tf

from constants import NUM_EPOCHS


class AbstractModelAttacker(abc.ABC):
    @abc.abstractmethod
    def forward_pass(self, dataset: tf.data.Dataset, server_model: tf.keras.Model):
        pass


class NoModelAttacker(AbstractModelAttacker):

    def __init__(self, model_fn: Callable[[], tf.keras.Model],
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss: tf.keras.losses.Loss) -> None:
        self.model_fn = model_fn
        self.optimizer = optimizer
        self.loss = loss

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
