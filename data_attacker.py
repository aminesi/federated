from abc import ABC, abstractmethod
from typing import List, Tuple, Set

import numpy as np
import tensorflow as tf


class AbstractDataAttacker(ABC):
    @abstractmethod
    def get_attacked_clients(self) -> Set[int]:
        pass

    @abstractmethod
    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        pass


class NoDataAttacker(AbstractDataAttacker):

    def get_attacked_clients(self) -> Set[int]:
        return set()

    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        return partitioned_data


class LabelAttacker(AbstractDataAttacker):

    def __init__(self, fraction: float) -> None:
        self.fraction = fraction
        self.attacked_clients = None

    def get_attacked_clients(self) -> Set[int]:
        return self.attacked_clients

    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        print(self.__class__.__name__ + " started.")
        classes = np.unique(np.concatenate([y for _, y in partitioned_data]))
        self.attacked_clients = np.random.choice(range(len(partitioned_data)), replace=False,
                                                 size=int(self.fraction * len(partitioned_data)))
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
