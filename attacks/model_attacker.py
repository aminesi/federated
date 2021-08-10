import abc
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from attacks.attacker import AbstractAttacker
from config import get_model, get_optimizer, get_loss
from utils.constants import NUM_EPOCHS, pick_clients


class AbstractModelAttacker(AbstractAttacker):

    def __init__(self, fraction: float) -> None:
        super().__init__(fraction)
        self.first_trainable_layer = 0

    @abc.abstractmethod
    def forward_pass(self, dataset: tf.data.Dataset, server_model: tf.keras.Model):
        pass

    def set_first_trainable_layer(self, first_trainable_layer):
        self.first_trainable_layer = first_trainable_layer

    def evaluate(self, test_data: tf.data.Dataset, server_model: tf.keras.Model):
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
        weights_delta = tf.nest.map_structure(tf.subtract, local_model.get_weights()[self.first_trainable_layer:],
                                              old_weights[self.first_trainable_layer:])
        return weights_delta


class RandomModelAttacker(AbstractModelAttacker):

    def __init__(self, fraction: float, std: float) -> None:
        super().__init__(fraction)
        self.std = std
        self.attacked_clients = set(pick_clients(fraction))

    def forward_pass(self, dataset: tf.data.Dataset, server_model: tf.keras.Model):
        return [np.random.normal(0, self.std, layer_weight.shape)
                for layer_weight in server_model.get_weights()[self.first_trainable_layer:]]


class SignFlipModelAttacker(NoModelAttacker):

    def __init__(self, fraction: float, multiplier: float) -> None:
        super().__init__()
        self.fraction = fraction
        self.multiplier = multiplier
        self.attacked_clients = set(pick_clients(fraction))

    def forward_pass(self, dataset: tf.data.Dataset, server_model: tf.keras.Model):
        clean_grads = super(SignFlipModelAttacker, self).forward_pass(dataset, server_model)
        return [layer * -1 * self.multiplier for layer in clean_grads]


class BackdoorAttack(NoModelAttacker):

    def __init__(self, fraction: float, target_class: int) -> None:
        super().__init__()
        self.fraction = fraction
        self.target_class = target_class
        self.pattern_relative_size = 0.15
        self.margin_relative_size = 0.05
        self.backdoor_test_samples = []
        self.chosen_attackers = 0

    def attack_train(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> \
            List[Tuple[np.ndarray, np.ndarray]]:
        print(self.__class__.__name__ + " started.")
        self.attacked_clients = pick_clients(self.fraction)
        for client in self.attacked_clients:
            x_train, y_train = partitioned_data[client]
            attacked_indices = self.pick_attacked_samples(y_train)
            y_train[attacked_indices] = self.target_class
            x_train[attacked_indices] = self.add_pattern((x_train[attacked_indices]))
            partitioned_data[client] = (x_train, y_train)
        self.attacked_clients = set(self.attacked_clients)
        print(self.__class__.__name__ + " finished.")
        return partitioned_data

    def attack_test(self, x_test, y_test):
        attacked_indices = self.pick_attacked_samples(y_test)
        y_test[attacked_indices] = self.target_class
        x_test[attacked_indices] = self.add_pattern((x_test[attacked_indices]))
        self.backdoor_test_samples = attacked_indices

    def add_pattern(self, x: np.ndarray):
        height, width, channel = x.shape[1:]
        pattern_width = int(np.round(self.pattern_relative_size * width))
        pattern_height = int(np.round(self.pattern_relative_size * height))
        left_margin = int(np.round(self.margin_relative_size * width))
        top_margin = int(np.round(self.margin_relative_size * height))
        slope = pattern_width / pattern_height
        x[:, [top_margin, top_margin + pattern_height], left_margin:left_margin + pattern_width + 1, :] = 255
        for image in x:
            for h in range(pattern_height):
                w = pattern_width - int(np.round(h / slope))
                image[top_margin + h, left_margin + w] = 255
        return x

    def pick_attacked_samples(self, labels: np.ndarray):

        pool = np.where(labels != self.target_class)[0]
        return np.random.choice(pool, min(int(np.round(.05 * len(labels))), len(pool)), False)

    def forward_pass(self, dataset: tf.data.Dataset, server_model: tf.keras.Model):
        print('back')
        weights_delta = super().forward_pass(dataset, server_model)

        # for epoch in range(NUM_EPOCHS):
        #     for x_batch, y_batch in dataset:
        #         with tf.GradientTape() as tape:
        #             out = local_model(x_batch, training=True)
        #             loss_val = self.loss(y_batch, out)
        #         grads = tape.gradient(loss_val, local_model.trainable_weights)
        #         self.optimizer.apply_gradients(zip(grads, local_model.trainable_weights))

        weights_delta = tf.nest.map_structure(lambda x: x * 10 / self.chosen_attackers, weights_delta)
        return weights_delta

    def evaluate(self, test_data: tf.data.Dataset, server_model: tf.keras.Model):
        x_test, y_test = test_data
        x_test = x_test[self.backdoor_test_samples]
        y_test = y_test[self.backdoor_test_samples]
        print('Backdoor accuracy: {}'.format(server_model.evaluate(x_test, y_test, verbose=0)[1]))
