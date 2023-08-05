from abc import ABC, abstractmethod
from typing import Any, List, Dict
from saturn.core.representations import Task

"""
    The base technique class. All other executors
    should subclass this abstract class. 
"""


class BaseTechnique(ABC):
    """
        The execute method takes in a task and a list of GPUs
        on which to run them. It should run the task to completion,
        then end without return.
    """
    @staticmethod
    @abstractmethod
    def execute(self, task: Task, gpu: List[int], tid: int, override_batch_count: int) -> None:
        pass

    """
        The autotuning function to optimize any technique-specific
        parameters. See FSDP.py and Pipeline.py for examples. Returns
        a dict with execution parameters and an estimated runtime. 
        Wrap in the @ray.remote wrapper.
    """
    @staticmethod
    @abstractmethod
    def search(self, task: Task, gpu: List[int], tid: int) -> Dict[str, Any]:
        pass
