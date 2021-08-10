from abc import abstractmethod
from typing import List, Tuple

import numpy as np

from attacks.attacker import AbstractAttacker
from utils.constants import pick_clients
from fed.partitioner import get_indices_by_class


class AbstractDataAttacker(AbstractAttacker):

    def __init__(self, fraction: float) -> None:
        super().__init__(fraction)

    @abstractmethod
    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        pass


class NoDataAttacker(AbstractDataAttacker):

    def __init__(self, fraction: float = 0) -> None:
        super().__init__(fraction)

    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        return partitioned_data


class LabelAttacker(AbstractDataAttacker):

    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        print(self.__class__.__name__ + " started.")
        classes = np.unique(np.concatenate([y for _, y in partitioned_data]))
        self.attacked_clients = pick_clients(self.fraction)
        for client in self.attacked_clients:
            y_list = np.array(partitioned_data[client][1])
            new_y_list = []
            for y in y_list:
                pool = np.delete(classes, y)
                new_y_list.append(np.random.choice(pool, 1, False)[0])
            new_y_list = np.array(new_y_list)
            partitioned_data[client] = (partitioned_data[client][0], new_y_list)
        self.attacked_clients = set(self.attacked_clients)
        print(self.__class__.__name__ + " finished.")
        return partitioned_data


class NoiseMutator(AbstractDataAttacker):

    def __init__(self, fraction: float, sigma_multiplier: float = 1) -> None:
        super().__init__(fraction)
        self.sigma_multiplier = sigma_multiplier

    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        print(self.__class__.__name__ + " started.")
        self.attacked_clients = pick_clients(self.fraction)
        for client in self.attacked_clients:
            x_train = partitioned_data[client][0]
            is_int8 = x_train.dtype == np.uint8
            x_train = x_train / 255
            for i, x in enumerate(x_train):
                std = np.std(x) * self.sigma_multiplier
                x = x + np.random.normal(0, std, x.shape)
                x = np.clip(x, 0, 1)
                x_train[i] = x
            x_train = x_train * 255
            if is_int8:
                x_train = np.round(x_train).astype(np.uint8)
            partitioned_data[client] = (x_train, partitioned_data[client][1])
        self.attacked_clients = set(self.attacked_clients)
        print(self.__class__.__name__ + " finished.")
        return partitioned_data


class DeleteMutator(AbstractDataAttacker):

    def __init__(self, fraction: float, delete_percentage: float = 0.75) -> None:
        super().__init__(fraction)
        self.delete_percentage = delete_percentage

    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        print(self.__class__.__name__ + " started.")
        self.attacked_clients = pick_clients(self.fraction)
        for client in self.attacked_clients:
            x_train, y_train = partitioned_data[client]
            grouped_indices = get_indices_by_class(y_train)
            remaining_indices = []
            for indices in grouped_indices:
                remaining_indices.append(
                    np.random.choice(indices, int(np.round((1 - self.delete_percentage) * len(indices))), False))
            remaining_indices = np.concatenate(remaining_indices)
            partitioned_data[client] = (x_train[remaining_indices], y_train[remaining_indices])
        self.attacked_clients = set(self.attacked_clients)
        print(self.__class__.__name__ + " finished.")
        return partitioned_data


class UnbalanceMutator(AbstractDataAttacker):

    def __init__(self, fraction: float, unbalance_percentage: float = 0.75) -> None:
        super().__init__(fraction)
        self.unbalance_percentage = unbalance_percentage

    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        print(self.__class__.__name__ + " started.")
        self.attacked_clients = pick_clients(self.fraction)
        for client in self.attacked_clients:
            x_train, y_train = partitioned_data[client]
            grouped_indices = get_indices_by_class(y_train)
            length_arr = [len(indices) for indices in grouped_indices]
            count_avg = np.mean(length_arr)
            chosen_index = -1
            if np.all(np.array(length_arr) == length_arr[0]):
                chosen_index = np.random.randint(0, len(grouped_indices))
            remaining_indices = []
            for i, indices in enumerate(grouped_indices):
                if len(indices) < count_avg or i == chosen_index:
                    remaining_indices.append(
                        np.random.choice(indices, int(np.round((1 - self.unbalance_percentage) * len(indices))), False))
                else:
                    remaining_indices.append(indices)
            remaining_indices = np.concatenate(remaining_indices)
            partitioned_data[client] = (x_train[remaining_indices], y_train[remaining_indices])
        self.attacked_clients = set(self.attacked_clients)
        print(self.__class__.__name__ + " finished.")
        return partitioned_data


class OverlapMutator(AbstractDataAttacker):

    def __init__(self, fraction: float, overlap_percentage: float = 0.75) -> None:
        super().__init__(fraction)
        self.overlap_percentage = overlap_percentage

    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        print(self.__class__.__name__ + " started.")
        self.attacked_clients = pick_clients(self.fraction)
        for client in self.attacked_clients:
            x_train, y_train = partitioned_data[client]
            grouped_indices = get_indices_by_class(y_train)
            length_arr = [len(indices) for indices in grouped_indices]
            if len(length_arr) < 2:
                continue
            group_index2, group_index1 = np.argsort(length_arr)[-2:]
            if np.all(np.array(length_arr) == length_arr[0]):
                group_index1, group_index2 = np.random.choice(range(len(grouped_indices)), 2, False)
            label2 = y_train[grouped_indices[group_index2][0]]
            indices = grouped_indices[group_index1]
            x = x_train[np.random.choice(indices, int(np.round(self.overlap_percentage * len(indices))), False)]
            y = np.full(len(x), label2)
            partitioned_data[client] = (np.vstack((x_train, x)), np.hstack((y_train, y)))
        self.attacked_clients = set(self.attacked_clients)
        print(self.__class__.__name__ + " finished.")
        return partitioned_data
