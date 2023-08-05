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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import saturn.core.executors.BaseTechnique

from enum import Enum
import ray


class Techniques(Enum):
    """Classification of training techniques.

    Can be extended in the future for more techniques.
    Currently supports, spilling, pipelining, FSDP, and Megatron-LM.
    """
    SPILLED = 1
    PIPELINE = 2
    FSDP = 3
    MEGATRON = 4


# @TODO add "estimated runtime" field

@ray.remote
def search(executor, task, gpus):
    return executor.search(task, gpus)

# @ray.remote


def execute(executor, task, gpus):
    return executor.execute(task, gpus)


class Strategy:
    """Training strategy representation.

    Combines a training technique with a resource apportionment.
    
    Parameters:
        executor: Which executor to use.
        
        gpu_apportionment: How many GPUs to execute with.
        
        parameters: Saved executor parameters.
        
        runtime: Estimated execution runtime.
    """

    def __init__(self, executor, gpu_apportionment: int, parameters: dict = None, runtime=None) -> None:

        if not isinstance(gpu_apportionment, int) or gpu_apportionment <= 0:
            raise ValueError("GPU allocation must be an integer > 0.")

        self.executor = executor
        self.gpu_apportionment = gpu_apportionment
        self.parameters = parameters
        self.runtime = runtime

    def __str__(self) -> str:
        return "Strategy({} ({}), {}G, {}s)".format(self.executor, self.parameters, self.gpu_apportionment, self.runtime)
