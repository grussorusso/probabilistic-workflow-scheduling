from numpy import random as rnd
import numpy as np

from scheduler.simulation import simulate
from scheduler.dyna_simulation import simulate_dyna
from scheduler.job import *
from scheduler.infrastructure import *
from scheduler.scheduling import SchedulingSolution, ScheduleEntry

def __evaluate (job, predictor, sol, task_durations):
    actual_schedules = {}

    to_complete = set(list(job.nodes))
    completion_times = {}
    last_completion_per_vm = {vm: 0.0 for vm in sol.vm_schedule}

    while len(to_complete) > 0:
        did_progress = False

        for vm in sol.vm_schedule:
            if len(sol.vm_schedule[vm]) == 0:
                continue
            next_subtask = sol.vm_schedule[vm][0].task

            can_execute = True
            start_time = last_completion_per_vm[vm]
            data_reading_time = 0
            for p in job.predecessors(next_subtask):
                if not p in completion_times:
                    # this subtask must wait
                    can_execute = False
                    break
                else:
                    start_time = max(start_time, completion_times[p])
                    if sol.subtask2instance[p] != vm:
                        data_reading_time = max(data_reading_time, predictor.data_reading_time(p[0],next_subtask[0],vm[0]))

            if not can_execute:
                continue

            completion_time = start_time + data_reading_time + task_durations[next_subtask]

            # check if we need to transfer output
            colocated_successors = True
            for p in job.successors(next_subtask):
                if sol.subtask2instance[p] != vm:
                    colocated_successors = False
            if not colocated_successors:
                    completion_time += predictor.data_writing_time(next_subtask[0], vm[0]) 

            if not vm in actual_schedules:
                actual_schedules[vm] = []
            entry = ScheduleEntry(next_subtask, start_time, completion_time)
            actual_schedules[vm].append(entry)
            
            completion_times[next_subtask] = completion_time
            last_completion_per_vm[vm] = completion_time
            to_complete.remove(next_subtask)
            sol.vm_schedule[vm] = sol.vm_schedule[vm][1:]

            did_progress = True
        if not did_progress:
            # likely a task dependency violation
            raise RuntimeError("There is likely a task dependency violation. Cannot simulate execution")

    makespan = max([completion_times[s] for s in completion_times])
    sol.vm_schedule = actual_schedules
    return makespan, sol


def evaluate_batch (n, job, predictor, sched_solution, rng=None,
                    accurate_simulation=True, billing_period_sec=0):
    makespans = np.zeros(n)
    costs = np.zeros(n)
    completed = np.zeros(n, dtype=int).astype(bool)

    # sample actual execution times
    all_task_durations = compute_task_durations_batch(n, job, sched_solution.vm_schedule, predictor, rng)

    for i in range(n):
        task_durations = {x: all_task_durations[x][i] for x in all_task_durations}
        if accurate_simulation:
            if sched_solution.ignore_instance_numbers:
                # Use Dyna-like simulator, which only considers the selected VM
                # types and schedules tasks to instances on-line
                if billing_period_sec <= 0:
                    billing_period_sec = 1
                makespan,sol = simulate_dyna(job, predictor, sched_solution, task_durations, billing_period_sec=billing_period_sec)
            else:
                makespan,sol = simulate(job, predictor, sched_solution, task_durations)
        else:
            makespan,sol = __evaluate(job, predictor, sched_solution, task_durations)

        if sol is None:
            # unfeasible run (likely a deadlock due to insufficient vCPU availability)
            completed[i] = False
        else:
            completed[i] = True
            makespans[i] = makespan
            costs[i] = evaluate_cost(sol, billing_period_sec)

    makespans = makespans[completed]
    costs = costs[completed]
    n_completed = np.sum(completed)

    return makespans,costs,n_completed


def compute_task_durations_batch (n, job, schedules, predictor, rng):
    durations = {}

    for vm in schedules:
        cold_start = True
        
        for entry in schedules[vm]:
            subtask = entry.task
            operator = subtask[0]
            
            if rng is None:
                avg_exec_time = predictor.exec_time (operator, job, vm[0], cold_start)
                durations[subtask] = [avg_exec_time for i in range(n)] # deterministic
            else:
                distribution = predictor.get_exec_time_distribution(operator, job, vm[0], cold_start)
                durations[subtask] = distribution.sample(rng, n)
            cold_start = False
    return durations

class MonteCarloResults:
    def __init__ (self, total_runs, makespans, costs, completed_count, deadline):
        self.avg_makespan = np.mean(makespans) if completed_count > 0 else -1
        self.avg_cost = np.mean(costs) if completed_count > 0 else -1
        self.std_makespan = makespans.std() if completed_count > 0 else -1
        self.std_cost = costs.std() if completed_count > 0 else -1
        self.violations = np.sum(makespans > deadline) + (total_runs-completed_count)
        violation_amount = makespans[makespans > deadline]-deadline
        if len(violation_amount) > 0:
            self.avg_tardiness = violation_amount.mean()
        else:
            self.avg_tardiness = 0
        self.hit_ratio = 1.0 - self.violations/total_runs if total_runs > 0  else 0
        self.unfeasible_runs = total_runs - completed_count
        self.total_runs = total_runs
        self.percentiles=[1,5,10,25,50,75,90,95,99]
        if completed_count > 0:
            self.makespan_quantiles = np.percentile(makespans, self.percentiles)
            self.cost_quantiles = np.percentile(costs, self.percentiles)
        else:
            self.makespan_quantiles = -1*np.ones(len(self.percentiles))
            self.cost_quantiles = -1*np.ones(len(self.percentiles))

    def __repr__ (self):
        return str(vars(self))

class MonteCarloEvaluator:

    def __init__ (self, job, predictor, deadline, deadline_percentile,
                  max_evaluations=-1, change_seed=True,
                  max_relative_error=0.02, verbose=True, batch_size=100, accurate_simulation=True,
                  billing_period=0):
        self.job = job
        self.predictor = predictor
        self.deadline = deadline
        self.deadline_percentile = deadline_percentile
        self.max_evaluations = max_evaluations
        self.max_relative_error = max_relative_error
        self.change_seed = change_seed
        self.verbose = verbose
        self.batch_size = batch_size
        self.accurate_simulation = accurate_simulation
        self.billing_period = billing_period

    def run (self, sol, initial_seed=123, detailed_results=False):
        if sol is None:
            if not detailed_results:
                return (0.0, -1.0, -1.0, -1.0)
            else:
                return MonteCarloResults(0, np.empty(0), np.empty(0), 0, self.deadline)


        sol = sol.copy() # we copy before sorting schedules
        sol.sort_schedules()

        violations = 0
        makespans = None
        costs = None
        completed_count = 0

        if not self.change_seed:
            eval_rng = rnd.default_rng(seed=initial_seed)

        i = 0
        while True:
            if self.change_seed:
                seed = initial_seed+i
                eval_rng = rnd.default_rng(seed=seed)

            _makespans, _costs, _completed = evaluate_batch(self.batch_size, self.job, self.predictor, sol, rng=eval_rng, accurate_simulation=self.accurate_simulation, billing_period_sec=self.billing_period)

            completed_count += _completed
            if makespans is None:
                makespans = _makespans
                costs = _costs
            else:
                makespans = np.concatenate((makespans, _makespans))
                costs = np.concatenate((costs, _costs))

            i += self.batch_size

            if completed_count == 0:
                # unfeasible
                break
            if self.max_evaluations > 0 and i >= self.max_evaluations:
                break

            # Check stopping criterion based on 95% confidence interval
            #https://quant.stackexchange.com/questions/21764/stopping-monte-carlo-simulation-once-certain-convergence-level-is-reached
            sigma = makespans.std()
            mu = makespans.mean()
            rel_error = sigma/math.sqrt(completed_count)*1.96/mu

            if rel_error < self.max_relative_error:
                break

        if not detailed_results:
            mean_makespan = np.mean(makespans) if completed_count > 0 else -1
            mean_cost = np.mean(costs) if completed_count > 0 else -1
            violations = np.sum(makespans > self.deadline) + (i-completed_count)
            hits = 1.0 - violations/completed_count if completed_count > 0  else 0

            if completed_count > 0:
                makespan_quantile = np.percentile(makespans, self.deadline_percentile*100)
            else:
                makespan_quantile = -1

            if self.verbose:
                print(f"MC summary: Sims:{i}; Makespan: {mean_makespan} (P{self.deadline_percentile*100}: {makespan_quantile}; Mean cost: {mean_cost}")

            return (hits, mean_makespan, mean_cost, makespan_quantile)
        else:
            results = MonteCarloResults(i, makespans, costs, completed_count, self.deadline)
            return results

def evaluate_cost (sched_solution: SchedulingSolution, billing_period_sec=0):
    cost = 0.0
    for vm in sched_solution.vm_schedule:
        if len(sched_solution.vm_schedule[vm])==0:
            continue
        start_time = min([e.est for e in sched_solution.vm_schedule[vm]])
        end_time = max([e.eft for e in sched_solution.vm_schedule[vm]])
        if billing_period_sec > 0:
            end_time = start_time + math.ceil(end_time/billing_period_sec)*billing_period_sec
        cost += vm[0].cost * (end_time - start_time)

    return cost
