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

from saturn.core.representations import Strategy
from saturn.library import retrieve
import torch
import ray
from timeit import default_timer as timer
import logging

@ray.remote
def ray_search(executor, task, gpu_range, tid):
    print("Starting Trial of Task: {}, Executor: {}, GPUs: {}".format(task.name, executor.name, ray.get_gpu_ids()))
    params, total_time = executor.search(task, gpu_range, tid)
    if params is not None:
        total_time = total_time * task.total_batches
    print("Finishing Trial of Task: {}, Executor: {}, GPUs: {}".format(task.name, executor.name, ray.get_gpu_ids()))
    return params, total_time


def search(tasks, executor_names=None, log=False):
    """
        Explores different combinations of jobs, execution strategies, and GPU counts
        to collect data needed for the plan solver.
        
        Parameters:
            tasks: A list of all tasks.

            executor_names: A list of executor strategy names to consider. These strategies should be registered
            in the library prior to search invocation. If set to None, then will use all strategies in the library
            by default.

            log: Determines whether or not to log execution details. Useful for debugging.

        Returns:
            No return. Strategy runtimes will be attached to the tasks directly.
    """
    
    if log:
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=logging.WARNING)
    
    if executor_names is None:
        executors = retrieve()
    else:
        executors = retrieve(executor_names)
    if not ray.is_initialized():
        context = ray.init(dashboard_host="0.0.0.0", num_gpus=torch.cuda.device_count(), resources={"node_0": 10000},
                           include_dashboard=True, configure_logging=True, logging_level='critical')  # just some arbitrarily high number
    flattened_results = []
    cpus_per_node = [(3 * int(x['Resources']['CPU']) // 4) for x in ray.nodes()]
    gpus_per_node = [int(x['Resources']['GPU']) for x in ray.nodes()]
    min_ratio_cpu_gpu_per_node = min([max(1, c//g) for c, g in zip(cpus_per_node, gpus_per_node)])
    max_gpus_per_node = max(gpus_per_node)
    default_gpu_range = [_ for _ in range(1, max_gpus_per_node+1)]
    ctr = 0
    
    for t in tasks:
        gpu_range = t.gpu_range
        if gpu_range is None:
            gpu_range = default_gpu_range
        for g in gpu_range:
            for exec in executors:
                g_range = [_ for _ in range(0, g)]
                ray_target = ray_search.options(
                    num_gpus=g, num_cpus=g*min_ratio_cpu_gpu_per_node).remote(exec, t, g_range, ctr)
                flattened_results.append(ray_target)
                ctr += 1
    
    # a magic number but generally found it's a reasonable upper bound...
    logging.info("{} trials to run. Expected wait time ~{}mins.".format(len(flattened_results), len(flattened_results)*1.2))
    st = timer()
    collected_flat_results = ray.get(flattened_results)
    end = timer()
    logging.info("Ran {} trials in {}hrs.".format(len(flattened_results), (end-st)/3600))
    res_per_task = len(gpu_range) * len(executors)
    for i in range(len(tasks)):
        task_results = collected_flat_results[i *
                                              res_per_task: (i+1) * res_per_task]
        for g_idx, g in enumerate(gpu_range):
            chosen_executor, chosen_parameters, chosen_runtime = None, None, None
            for exec_idx, exec in enumerate(executors):
                (params,
                    runtime) = task_results[g_idx * len(executors) + exec_idx]
                if params is not None:
                    if chosen_runtime is None or runtime < chosen_runtime:
                        chosen_executor = exec
                        chosen_parameters = params
                        chosen_runtime = runtime
            tasks[i].strategies[g] = Strategy(
                chosen_executor, g, chosen_parameters, chosen_runtime)
