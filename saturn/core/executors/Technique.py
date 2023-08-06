# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
    name = "BaseTechnique (override when extending)"
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
