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

from saturn.solver import solve, convert_into_comprehensible
from saturn.executor import forecast, execute
import ray
import logging
import os


@ray.remote(num_cpus=max(1, os.cpu_count() // 4))
def ray_solve(task_list, presolved=None):
    return solve(task_list, presolved)


"""
    Orchestrates execution, including introspection
    with overlapping solve/execute phases.
"""


def orchestrate(task_list, log=False, interval=1000):
    """
        Primary entry-point for job submission and execution.
        
        Parameters:
            task_list: A list of all tasks.

            log: Whether or not to log execution details.

            interval: Introspection interval granularity. Defaults to 1000, can be modified to affect performance.

        Returns:
            No return. Trained models will be saved to their specified locations.
        """
    
    if log:
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=logging.WARNING)
    
    INTERVAL = interval
    res = ray_solve.remote(task_list)
    presolved = ray.get(res)  # initial pass, need MILP up front
    interval_sta, interval_tga, interval_bss, interval_bna, interval_boa, saved_makespan = presolved
    

    npt, tdd, sta_comp = convert_into_comprehensible(
        task_list, interval_bss, interval_boa, interval_tga, interval_bna, interval_sta)


    while len(task_list) > 0:
        rtt, btr, cmp = forecast(task_list, INTERVAL, sta_comp)
        logging.info("Launching {} in this interval.".format([t.name for t in rtt]))
        logging.info("Forecasting that {} will finish in this interval.".format([t.name for t in cmp]))
        task_list = [t for t in task_list if t not in cmp]
        res = ray_solve.remote(task_list, presolved)
        execute(rtt, btr, INTERVAL, npt, tdd)
        presolved = ray.get(res)
        interval_sta, interval_tga, interval_bss, interval_bna, interval_boa, saved_makespan = presolved
        npt, tdd, sta_comp = convert_into_comprehensible(
            task_list, interval_bss, interval_boa, interval_tga, interval_bna, interval_sta)
