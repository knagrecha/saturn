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

"""Contains all execution-related functionality.
"""

import ray
from timeit import default_timer as timer
import logging
import time
import asyncio


@ray.remote
class DependencyHolder:
	def __init__(self, relevant_tasks):
		self.completed_tasks = [asyncio.Event() for _ in relevant_tasks]
		for c in self.completed_tasks:
			c.clear()
	
	# self.in_progress_or_completed_tasks = [0 for _ in relevant_tasks]
	
	def set_task_complete(self, tid):
		self.completed_tasks[tid].set()
	
	async def wait(self, tid, should_wait=True):
		if should_wait:
			await self.completed_tasks[tid].wait()
	
	def get_in_progress_array(self):
		return self.in_progress_or_completed_tasks
	
	def get_completed_array(self):
		return self.completed_tasks
	
	def set_task_started(self, tid):
		self.in_progress_or_completed_tasks[tid] = 1


@ray.remote
class LauncherActor:
	def __init__(self, global_dependency_holder):
		self.global_dependency_holder = global_dependency_holder
	
	def ray_execute(self, task, batch_count, tid, task_dep_indices, ratio_cpu_gpu, npt_r):
		for dep_tid in task_dep_indices:
			ray.get(self.global_dependency_holder.wait.remote(dep_tid))
		
		exec_actor = ExecutorActor.options(
			num_cpus=task.selected_strategy.gpu_apportionment * ratio_cpu_gpu,
			num_gpus=task.selected_strategy.gpu_apportionment,
			resources={"node_{}".format(npt_r): 1}).remote()
		
		ray.get(exec_actor.run.remote(task, batch_count, tid))
		ray.kill(exec_actor)
		ray.get(self.global_dependency_holder.set_task_complete.remote(tid))
		print("Marking task {} as completed for the current interval.".format(task.name))


@ray.remote
class ExecutorActor:
	def __init__(self):
		pass
	
	def run(self, task, batch_count, tid):
		"""Only used internally. Launched by the execute function to submit Ray jobs.
        """
		
		executor = task.selected_strategy.executor
		print("Marking task {} as started with executor {} on {}.".format(task.name, executor, ray.get_gpu_ids()))
		# task.setup()
		executor.execute(task, [_ for _ in range(
			0, task.selected_strategy.gpu_apportionment)], tid, batch_count)
		task.reconfigure(batch_count)
		


def execute(relevant_tasks, batches_to_run, interval, node_per_task, task_dependency_dict):
	"""Executes a list of tasks for the given interval.
        
        Parameters:
            relevant_tasks: A list of the tasks to be submitted in the current interval.
            
            batches_to_run: A list of integers, each detailing how many batches to run each relevant task for in the current interval.
            
            interval: Interval length in seconds.
            
            node_per_task: Which node to run each task on.
            
            task_dependency_dict: Dictionary used to determine schedule orderings.
        
        Returns:
            None
    """
	task_deps = DependencyHolder.remote(relevant_tasks)
	
	cpus_per_node = [(3 * int(x['Resources']['CPU']) // 4) for x in ray.nodes()]
	gpus_per_node = [int(x['Resources']['GPU']) for x in ray.nodes()]
	
	st = timer()
	
	task_exec_list = [None for _ in relevant_tasks]
	
	for r_idx, r in enumerate(relevant_tasks):
		# check if all dependencies resolved
		task_dep_indices = [relevant_tasks.index(t) for t in task_dependency_dict[r]]
		ratio_cpu_gpu = max(1, cpus_per_node[node_per_task[r]] // gpus_per_node[node_per_task[r]])
		launcherActor = LauncherActor.remote(task_deps)
		task_exec_list[r_idx] = launcherActor.ray_execute.remote(r, batches_to_run[r_idx], r_idx, task_dep_indices, ratio_cpu_gpu,
		                                 node_per_task[r])
	
	logging.info("ALL JOBS LAUNCHED FOR THIS INTERVAL")
	ray.get(task_exec_list)
	end = timer()
	logging.info("Intended Interval Time: {}, Actual Time: {}".format(interval, end - st))
	if end - st > interval:
		logging.info("Underestimated runtime by {}%.".format(((end - st) / interval - 1) * 100))
	else:
		logging.info("Overestimated runtime by {}%.".format((interval / (end - st) - 1) * 100))


def forecast(task_list, interval, interval_sta):
	"""
        Uses strategy runtimes and estimated start times to determine which jobs
        run in the current interval, and how many batches they should run.
        We use this to plan graceful interval termination, rather than using interruption.
        
        Parameters:
            task_list: A list of all tasks.
            
            interval: The interval runtime.
            
            interval_sta: A list of start times for each job.
        
        Returns:
            (relevant_tasks, batches_to_run, completed_tasks)
            
            relevant_tasks: Tasks that must run in the upcoming interval.
            
            batches_to_run: How many batches each of those tasks should run for.
            
            completed_tasks: Any tasks that have been fully completed.
        
    """
	
	relevant_tasks = [task_list[idx]
	                  for idx, st in enumerate(interval_sta) if st < interval]
	expected_total_time_per_task = [t.selected_strategy.runtime for t in relevant_tasks]
	time_in_interval_per_task = [interval -
	                             st for st in interval_sta if st < interval]
	batch_time_per_task = [ex / t.total_batches for ex, t in zip(expected_total_time_per_task, relevant_tasks)]
	batches_to_run = [min(ta.total_batches, t // b)
	                  for ta, t, b in zip(relevant_tasks, time_in_interval_per_task, batch_time_per_task)]
	
	completed_tasks = set()
	for r_idx, r in enumerate(relevant_tasks):
		for g_count, strat in r.strategies.items():
			r.strategies[g_count].runtime -= max(0, (strat.runtime /
			                                         r.total_batches) * batches_to_run[r_idx])
		
		# reduce the number of total batches to run
		r.total_batches = max(0, r.total_batches - batches_to_run[r_idx])
		if r.total_batches <= 0:
			completed_tasks.add(r)
			logging.info("Task {} will finish entirely in the current interval. Model file will be saved to the "
			             ".models/ directory.".format(r.name))
	
	return relevant_tasks, batches_to_run, completed_tasks
