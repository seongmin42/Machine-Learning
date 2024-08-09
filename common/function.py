from abc import ABC, abstractmethod


class Function(ABC):
    @abstractmethod
    def calc(self, *args):
        pass
