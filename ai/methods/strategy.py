from abc import ABC, abstractmethod

class ArchitectureAnalysisStrategy(ABC):

    @abstractmethod
    def run(self, *args, **kwargs):
        pass