from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import tensorflow as tf


class BaseAttacker(ABC):
    @abstractmethod
    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        pass


class NoAttacker(BaseAttacker):
    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        return partitioned_data


class LabelFlipping(BaseAttacker):

    def __init__(self, fraction: float) -> None:
        self.fraction = fraction

    def attack(self, partitioned_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        print(self.__class__.__name__ + " started.")
        classes = np.unique(np.concatenate([y for _, y in partitioned_data]))
        high = len(classes)
        for client in np.random.choice(range(len(partitioned_data)), replace=False,
                                       size=int(self.fraction * len(partitioned_data))):
            y = np.array(partitioned_data[client][1])
            rand = np.random.randint(low=0, high=high, size=len(y)).astype(y.dtype)
            partitioned_data[client] = (partitioned_data[client][0], tf.convert_to_tensor(rand))
        print(self.__class__.__name__ + " finished.")
        return partitioned_data
