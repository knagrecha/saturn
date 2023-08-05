from pulp import *
import pulp
import numpy as np
import torch
from saturn.core.representations import Task
from typing import List
from collections import defaultdict
import ray
import os




def solve(task_list: List[Task], presolved=None, gurobi=True):
    """
        The MILP-based solver at the core of Saturn. Currently relies on Gurobi
        for plan construction, but we are actively looking into alternatives that do not
        require external dependencies. For now, if you do not have access to a Gurobi license,
        set the gurobi parameter to False. In this case, Saturn will use the open-source
        CBC solver from PuLP. Note that this is generally significantly slower than
        Gurobi, so your runtimes & query execution plan quality may suffer as a result.
        
        Parameters:
            task_list: A list of all tasks.

            presolved: Used for warm-starting in our introspection/interval-based solving scheme.

            gurobi: Whether or not to use the Gurobi solver. If set to false, will use a slower, open-source alternative.
            
        Returns:
            interval_sta: Start time array for the current interval.
            
            interval_tga: Task GPU allocation array for the current interval.
            
            interval_bss: Binary strategy selection array for the current interval.
            
            interval_bna: Binary node selection array for the current interval.
            
            interval_boa: Before or After array to determine schedule orderings for the current interval.
            
            saved_makespan: Float representing expected runtime for the current interval.
    """
    
    if not ray.is_initialized():
        context = ray.init(resources={"node_0": 10000}, configure_logging=False, logging_level="critical")  # just some arbitrarily high number

    DEBUG = True
    NODES = len(ray.nodes())
    if not DEBUG:
        gpus_per_node = [int(x['Resources']['GPU']) for x in ray.nodes()]
    else:
        gpus_per_node = [8 for x in ray.nodes()]


    """
        Setup phase
    """
    TASK_COUNT = len(task_list)
    node_gpu_array = gpus_per_node

    gpu_time_tuples = []
    
    """"
        As of Python 3.7, insertion order is guaranteed when accessing dict items.
        So we can be assured that the ordering is consistent.
    """
    
    for task in task_list:
        curr_gpu_tuples = []
        for g_count, strat in task.strategies.items():
            curr_gpu_tuples.append((g_count, strat.runtime))
        gpu_time_tuples.append(curr_gpu_tuples)

    """
        Invoked if Gurobi cannot be installed onto the local machine (e.g. for licensing reasons).
        Will make a GET request to a user-specified machine address for solving.

    """


    def create_and_solve_prob(gpu_time_tuples, TASK_COUNT, makespan_opt=True, bss=None, sta=None, tga=None):
        prob = LpProblem('Scheduling Makespan', LpMinimize)

        """
            Strategy Selection
        """

        binary_strategy_selection_array = []
        for t in range(TASK_COUNT):
            binary_task_strategy_selection_array = []
            for strat_idx, strat in enumerate(gpu_time_tuples[t]):
                binary_task_strategy_selection_array.append(LpVariable("BinarySelectedStrategy_Task_{}_Strategy_{}".format(t, strat_idx),
                                                                       cat="Binary"))
                if bss is not None:
                    binary_task_strategy_selection_array[-1].setInitialValue(
                        bss[t][strat_idx])

            binary_strategy_selection_array.append(
                binary_task_strategy_selection_array)

        # ONLY ONE STRATEGY PER TASK

        for strat in binary_strategy_selection_array:
            prob += lpSum(strat) == 1.0

        """
            Node Selection
        """

        binary_node_selection_array = []

        for t in range(TASK_COUNT):
            binary_task_node_selection_array = []
            for node in range(NODES):
                binary_task_node_selection_array.append(LpVariable("BinarySelectedNode_Task_{}_Node_{}".format(t, node),
                                                                   cat="Binary"))

            binary_node_selection_array.append(
                binary_task_node_selection_array)

        """
            For each task, an array of 0s or 1s saying which node is selected

            binary_node_selection_array[t][n] = was task t on node n?
        """

        # ONLY ONE NODE PER TASK

        for node in binary_node_selection_array:
            prob += lpSum(node) == 1.0

        start_time_array = [
            [
                [
                    LpVariable("StartTime_Node_{}_GPU_{}_task_{}".format(n, g, t),
                               lowBound=0, cat="Integer")
                    for t in range(TASK_COUNT)
                ]
                for g in range(node_gpu_array[n])
            ]
            for n in range(NODES)
        ]

        if sta is not None:
            for n in range(NODES):
                for g in range(node_gpu_array[n]):
                    for t in range(TASK_COUNT):
                        start_time_array[n][g][t].setInitialValue(max(0, sta[n][g][t]))

        """

            array describes [N][G][T] = start time of task T on Node N, GPU G of Node N
        """

        makespan = LpVariable("Makespan", 0)
        M = 10000000

        """
            Makespan must be higher than start time + time of the selected strategy
        """
        per_task_completion_time = [LpVariable(
            "CompletionTime_task_{}".format(t)) for t in range(TASK_COUNT)]
        for node_idx, node in enumerate(start_time_array):
            for gpu_idx, gpu in enumerate(node):
                for t_idx, time_st in enumerate(gpu):
                    for s_idx, strategy in enumerate(binary_strategy_selection_array[t_idx]):
                        if makespan_opt:
                            prob += makespan >= time_st + \
                                gpu_time_tuples[t_idx][s_idx][1] - \
                                (M * (1-strategy))
                        else:
                            prob += per_task_completion_time[t_idx] >= time_st

        if not makespan_opt:
            prob += makespan >= lpSum(per_task_completion_time)

        task_gpu_occupancy_array = [
            [
                [
                    LpVariable("NODE_{}_GPU_{}_Task{}".format(
                        n, g, t), cat="Binary")

                    for g in range(node_gpu_array[n])
                ]
                for n in range(NODES)
            ]
            for t in range(TASK_COUNT)
        ]

        if tga is not None:
            for t in range(TASK_COUNT):
                for n in range(NODES):
                    for g in range(node_gpu_array[n]):
                        task_gpu_occupancy_array[t][n][g].setInitialValue(
                            tga[t][n][g])

        """
            array describes [T][N][G] = was task T running on GPU G of Node N?
        """

        # GPU count constraint
        for task_idx, task in enumerate(task_gpu_occupancy_array):
            for node_idx, node in enumerate(task):

                # if true, this = 0. Otherwise = 1
                on_node = (
                    1-binary_node_selection_array[task_idx][node_idx]) * M
                # if on node, this = M. Otherwise = 0
                off_node = binary_node_selection_array[task_idx][node_idx] * M

                for s_idx, strategy in enumerate(binary_strategy_selection_array[task_idx]):

                    on_strategy = (1-strategy) * M

                    prob += lpSum(
                        node) >= gpu_time_tuples[task_idx][s_idx][0] - on_strategy - on_node
                    prob += lpSum(
                        node) <= gpu_time_tuples[task_idx][s_idx][0] + on_strategy + on_node

                    prob += lpSum(node) >= 0 - off_node
                    prob += lpSum(node) <= 0 + off_node

        """
        Start time consistency across GPUs
        """

        for task in range(TASK_COUNT):

            for node_idx, node in enumerate(task_gpu_occupancy_array[task]):

                # if true, this = 0. Otherwise = 1
                on_node = (1-binary_node_selection_array[task][node_idx]) * M

                start_times_for_task = [g[task]
                                        for g in start_time_array[node_idx]]

                # lpSum(start_times_for_task) / strategy count
                # then turn the constraint on and off depending on if that strategy was selected

                for strat_idx, strat in enumerate(gpu_time_tuples[task]):
                    target = lpSum(start_times_for_task) / strat[0]
                    strat_selected = (
                        1-binary_strategy_selection_array[task][strat_idx]) * M

                    for gpu_idx, gpu in enumerate(node):
                        # if not on that gpu (gpu == 0), pass by default
                        prob += target <= start_time_array[node_idx][gpu_idx][task] + (
                            (1-gpu) * M) + strat_selected + on_node
                        prob += target >= start_time_array[node_idx][gpu_idx][task] - (
                            (1-gpu) * M) - strat_selected - on_node

        """
            Worker exclusion

        """

        before_or_after = [
            [
                (LpVariable("IsTask{}_AfterTask{}".format(
                    w1, w2), cat="Binary"))
                for w1 in range(TASK_COUNT)
            ]
            for w2 in range(TASK_COUNT)
        ]

        """
            BoA[t1][t2] describes if t2 was after t1

        """

        for node_idx, node in enumerate(start_time_array):
            for gpu_idx, gpu in enumerate(node):

                for task_idx, start_time in enumerate(gpu):

                    for task_idx_prime, start_time_prime in enumerate(gpu):

                        if task_idx_prime == task_idx:
                            continue

                        did_run_prime = (
                            1-task_gpu_occupancy_array[task_idx_prime][node_idx][gpu_idx]) * M
                        did_run = (
                            1-task_gpu_occupancy_array[task_idx][node_idx][gpu_idx]) * M

                        after = before_or_after[task_idx_prime][task_idx] * M
                        before = (
                            1-before_or_after[task_idx_prime][task_idx]) * M

                        """
                            task_idx_prime ran after task_idx

                        """

                        for strat_idx, strat in enumerate(gpu_time_tuples[task_idx]):

                            # CURRENT BEFORE PRIME
                            strat_selected = (
                                1-binary_strategy_selection_array[task_idx][strat_idx])*M
                            prob += start_time <= start_time_prime - \
                                strat[1] + did_run_prime + \
                                did_run + after + strat_selected

                        """
                            task_idx_prime ran before task_idx

                        """

                        for strat_idx, strat in enumerate(gpu_time_tuples[task_idx_prime]):

                            # CURRENT AFTER PRIME
                            strat_selected = (
                                1-binary_strategy_selection_array[task_idx_prime][strat_idx])*M
                            prob += start_time >= start_time_prime + \
                                strat[1] - did_run - did_run_prime - \
                                before - strat_selected

        prob.setObjective(makespan)
        if gurobi:
            solver = GUROBI_CMD(timeLimit=500, threads=os.cpu_count() // 4, warmStart=True)
        else:
            solver = PULP_CBC_CMD(timeLimit=500, threads=os.cpu_count() // 4, warmStart=True)
	        
        prob.solve(solver)
        

        sta = [[[t.value() for t in g] for g in n]
                            for n in start_time_array]
        
        tga = [[[g.value() for g in n] for n in t]
                        for t in task_gpu_occupancy_array]
        bss = [[s.value() for s in t] for t in binary_strategy_selection_array]
        bna = [[n.value() for n in t] for t in binary_node_selection_array]
        boa = [[t_prime.value() for t_prime in t] for t in before_or_after]


        # Return problem and decision variables
        return (
            bss,
            bna,
            sta,
            tga,
            boa,
            makespan.value()

        )

    INTERVAL = 1000
    saved_makespan = None
    interval_sta = None
    interval_tga = None
    interval_bss = None
    interval_bna = None
    interval_boa = None
    if presolved is not None:
        interval_sta, interval_tga, interval_bss, interval_bna, interval_boa, saved_makespan = presolved

    def introspection_wrapper(current_makespan, threshold=500, interval=1000):
      
        (
            bss,
            bna,
            sta,
            tga,
            boa,
            makespan
        ) = create_and_solve_prob(gpu_time_tuples, TASK_COUNT, makespan_opt=True,
                                bss=interval_bss, sta=interval_sta, tga=interval_tga)

        swap = False
        if current_makespan is not None:
            swap = makespan < saved_makespan - interval - threshold
        else:
            swap = True

            

        return swap, bss, bna, sta, tga, boa, makespan

    def solution_comparator():
        nonlocal saved_makespan
        nonlocal interval_sta
        nonlocal interval_tga
        nonlocal interval_bss
        nonlocal interval_bna
        nonlocal interval_boa
        swap, prop_bss, prop_bna, prop_sta, prop_tga, prop_boa, prop_makespan = introspection_wrapper(
            saved_makespan)
        
        # NO PLAN CURRENTLY EXISTS or TASKS COMPLETED IN PREVIOUS INTERVAL
        if saved_makespan is None or len(interval_tga) > len(prop_tga):
            interval_sta = prop_sta
            interval_tga = prop_tga
            interval_bss = prop_bss
            interval_bna = prop_bna
            interval_boa = prop_boa
            
        elif swap: # PLAN EXISTS, BUT WE WANT TO SWAP
            """
                Swap over the arrays
            """
            for n_idx, n in enumerate(prop_sta):
                for g_idx, g in enumerate(n):
                    for t_idx, t in enumerate(g):
                        interval_sta[n_idx][g_idx][t_idx] = t

            for t_idx, t in enumerate(prop_tga):
                for n_idx, n in enumerate(t):
                    for g_idx, g in enumerate(n):
                        interval_tga[t_idx][n_idx][g_idx] = g

            for t_idx, t in enumerate(prop_bss):
                for s_idx, s in enumerate(t):
                    interval_bss[t_idx][s_idx] = s

            for t_idx, t in enumerate(prop_bna):
                for n_idx, n in enumerate(t):
                    interval_bna[t_idx][n_idx] = n

            for t_idx, t in enumerate(prop_boa):
                for t_prime_idx, t_prime in enumerate(t):
                    interval_boa[t_idx][t_prime_idx] = t_prime

            saved_makespan = prop_makespan
        
        else: # CONTINUE WITH CURRENT PLAN
            """
                @TODO: eliminate unnecessary interruptions when schedule simply continues
            """

            saved_makespan = saved_makespan - INTERVAL

            """
                Update start times (start has moved forwrad by INTERVAL)
            """
            for n_idx, n in enumerate(interval_sta):
                for g_idx, g in enumerate(n):
                    for t_idx, t in enumerate(g):
                        interval_sta[n_idx][g_idx][t_idx] = max(
                            t-INTERVAL, 0)

    solution_comparator()
    return interval_sta, interval_tga, interval_bss, interval_bna, interval_boa, saved_makespan





def convert_into_comprehensible(task_list, bss, boa, tga, bna, sta):
    """
        Converts MILP outputs into a format somewhat easier to work with.

        Parameters:
            task_list: A list of all tasks.

            bss: Binary Strategy Search array. Indicates which strategy is used by each task.

            boa: Before or After array. Indicates whether a task ran before or after another one; used for
            dependency graph construction in scheduling.
            
            tga: Task GPU allocation array. Determines which GPUs each task will block.
            
            bna: Binary Node Array. Determines which node each task will run on.
            
            sta: Start Time Array. Determines the start time of each task.

        Returns:
            No return. Strategy runtimes will be attached to the tasks directly.
    """
    
    # which node for each task?
    node_per_task = {}
    for idx, task in enumerate(task_list):
        node_per_task[task] = np.argmax(bna[idx])

        
    for idx, task in enumerate(task_list):
        # bss is easy - simply tells us which strategy to use
        ctr = 0
        """
            Exploit consistent insertion order for dictionaries
        """
        for strat_key, strat in task.strategies.items():
            if ctr == np.argmax(bss[idx]):
                task.select_strategy(strat)
                logging.info("Task {} wil select strategy {}".format(task.name, strat))
                break
            else:
                ctr += 1


    # list of tasks that need to be resolved before launching this job
    task_dependency_dict = defaultdict(list)
    start_time_per_task = []

    for idx, task in enumerate(task_list):
        blocked_node = tga[idx][node_per_task[task]]
        blocked_gpus = [g_idx for g_idx, g in enumerate(blocked_node) if g == 1]

        logging.info("Task {} will run on node {}".format(task.name, node_per_task[task]))
        logging.info("Task {} will block GPUs {}".format(task.name, blocked_gpus))
        
        start_time_per_task.append(sta[node_per_task[task]][blocked_gpus[0]][idx])

        for idx_prime, task_prime in enumerate(task_list):
            if idx_prime == idx:
                continue

            if node_per_task[task_prime] != node_per_task[task]:
                continue

            blocked_node_prime = tga[idx_prime][node_per_task[task]]
            blocked_gpus_prime = [g_idx for g_idx,
                                  g in enumerate(blocked_node_prime) if g == 1]

            if (set(blocked_gpus).intersection(blocked_gpus_prime)):
                if boa[idx_prime][idx]:  # if task_prime was before task
                    task_dependency_dict[task].append(task_prime)
                    logging.info("Task {} will wait on task {}".format(task.name, task_prime.name))

    return node_per_task, task_dependency_dict, start_time_per_task
