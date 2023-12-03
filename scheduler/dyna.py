import random
import os
import copy
import networkx as nx
import heapq
import time

from scheduler.job import *
from scheduler.infrastructure import *
from scheduler.scheduling import SchedulingSolution, ScheduleEntry
from scheduler import evaluation as monte_carlo


def get_mostexpensive_slowest_machine(machines):
    i=0
    max_cost=0
    min_cost=0
    for i in range(len(machines)):
        if machines[i].cost > machines[max_cost].cost:
            max_cost = i
        if machines[i].cost < machines[min_cost].cost:
            min_cost= i
    return max_cost,min_cost
# ----------------------------------------- sim.py


def make_scheduling_solution (solution, topo_sort):
    # Maps each VM type to the next instance number
    vmtype2instances = {}
    vm_schedule = {}
    task2vm = {}

    for i,vmt in enumerate(solution):
        vmtype2instances[vmt] = vmtype2instances.get(vmt, -1) + 1
        vm = (vmt, vmtype2instances[vmt])
        task     = topo_sort[i]
        vm_schedule[vm] = [ScheduleEntry(task, 0)]
        task2vm[task] = vm

    sched_solution = SchedulingSolution()
    sched_solution.subtask2instance = task2vm
    sched_solution.vm_schedule = vm_schedule

    return sched_solution


def evaluate(solution, mc_evaluator, topo_sort):
    a,b,c,d = mc_evaluator.run(make_scheduling_solution(solution, topo_sort))
    return (solution,a,b,c,d)



class DynaScheduler:

    def __init__ (self, infra, pred, max_iterations, deadline_percentile=0.9,
                  max_execution_sec=3600, mc_stopping_rel_error=0.1):
        self.infrastructure = infra
        self.predictor = pred
        self.deadline_percentile = deadline_percentile
        self.max_iterations = max_iterations
        self.max_execution_sec = max_execution_sec
        self.parallel = False
        self.mc_stopping_rel_error = mc_stopping_rel_error

        if self.parallel:
           import dask
           from dask.distributed import Client
           from distributed import LocalCluster
           dask.config.set(scheduler='processes')
           self.client = Client(n_workers=20, threads_per_worker=1)
           self.ncores = sum(self.client.ncores().values())
           print(self.ncores)

    def schedule (self, job, deadline):
        assert(isinstance(job, Job))
        assert(deadline > 0.0)
        all_topo_sorts = nx.all_topological_sorts(job)
        topological_sort = next(all_topo_sorts)
        vm_types = list(self.infrastructure.vm_types)
        _,self.min_power = get_mostexpensive_slowest_machine(vm_types)
        self.min_power = vm_types[self.min_power]
        num_tasks = len(job.nodes)

        mc_evaluator = monte_carlo.MonteCarloEvaluator(job, self.predictor,
                                                       deadline, self.deadline_percentile,
                                                       max_relative_error=self.mc_stopping_rel_error,
                                                       change_seed=False,
                                                       verbose=False,
                                                       batch_size=100,
                                                       accurate_simulation=False)

        closed_list = set([])
        open_list     = []
        iterations = 0
        upper_bound = float("inf") 
 
        initial_solution = tuple([self.min_power]*num_tasks)
        result = initial_solution
        initial_solution,_,_,f_cost,makespan_perc = evaluate(initial_solution, mc_evaluator, topological_sort)
        f_cost *= 2
        level  = -1
        heapq.heappush(open_list,(f_cost,iterations,makespan_perc,level,initial_solution))

        start_time = time.time()

        if self.parallel:
            import dask
            evaluate_solution = dask.delayed(evaluate)
    
        while len(open_list) > 0 and iterations < self.max_iterations:
            iterations+=1
            if iterations % 100 == 0:
                print(f"Iterations: {iterations}")
            res = heapq.heappop(open_list)
            f_cost,_,makespan,level,current_solution = res
            if makespan <= deadline and f_cost < upper_bound:
                result = current_solution
                upper_bound = f_cost
            closed_list.add((current_solution,level))
            level+=1
            if level < len(job.nodes()):
                parent = list(current_solution)

                future = []
                for vm_type in vm_types:
                    parent[level] = vm_type
                    neighbor = tuple(parent)
                    if (neighbor,level) in closed_list:
                        continue
                    if neighbor != tuple(current_solution):
                        if self.parallel:
                            new_evaluated_solution = evaluate_solution(neighbor, mc_evaluator, topological_sort)
                        else:
                            new_evaluated_solution = evaluate(neighbor, mc_evaluator, topological_sort)
                        future.append(new_evaluated_solution)
                    else:
                        n_cost = f_cost
                        n_msp  = makespan
                        if n_cost > upper_bound:
                            continue
                        heapq.heappush(open_list,(n_cost,iterations,n_msp,level,neighbor))

                if self.parallel:
                    import dask
                    future = dask.compute(*future)
                    
                for neighbor,_,_,n_cost,n_msp in future:
                    if n_cost > upper_bound:
                        continue
                    heapq.heappush(open_list,(n_cost*2,iterations,n_msp,level,neighbor))

            if self.max_execution_sec > 0 and time.time()-start_time > self.max_execution_sec:
                break


        print("Iterations:",iterations)
        print("Found solution:",result)

        sched_sol = make_scheduling_solution(result, topological_sort)
        if sched_sol is not None:
            sched_sol.ignore_instance_numbers = True # online instance scheduling
        return sched_sol

