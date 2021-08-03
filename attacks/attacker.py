from abc import ABC
from typing import Set


class AbstractAttacker(ABC):

    def __init__(self, fraction: float) -> None:
        self.fraction = fraction
        self.attacked_clients = set()

    def get_attacked_clients(self) -> Set[int]:
        return self.attacked_clients
