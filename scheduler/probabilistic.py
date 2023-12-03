from scipy.stats import gamma
import time
import random
import math
import heapq
import numpy as np
import os
import functools
from multiprocessing import Pool

from scheduler.job import *
from scheduler.infrastructure import *
from scheduler.scheduling import SchedulingSolution, ScheduleEntry
from scheduler.heft import CloudMOHEFT
from scheduler.evaluation import MonteCarloEvaluator

class ProbabilisticMOHEFT ():

    def __init__ (self, infra, pred, required_percentile=0.9, K=3,
                  return_frontier=False,
                  mc_stopping_rel_error=0.1,
                  conservative_hit_ratio_margin=0.0,
                  accurate_monte_carlo=False,
                  percentile_stopping_threshold=0.02):
        self.base_predictor = pred
        assert(required_percentile >= 0.0)
        assert(required_percentile <= 1.0)
        self.required_percentile = required_percentile
        self.K = K
        self.infra = infra
        self.return_frontier = return_frontier
        self.percentile_stopping_threshold = percentile_stopping_threshold
        self.mc_stopping_rel_error = mc_stopping_rel_error
        self.conservative_hit_ratio_margin = min(conservative_hit_ratio_margin,
                                                 1.0-self.required_percentile)
        self.accurate_monte_carlo=accurate_monte_carlo

    def schedule (self, job, deadline):
        p0 = 0.5
        p1 = 0.99
        feasible_solutions = []
        all_solutions = []

        total_mc = 0
        total_mohe = 0

        mc = MonteCarloEvaluator(job, self.base_predictor, deadline,
                                 self.required_percentile, 
                                 max_relative_error=self.mc_stopping_rel_error,
                                 change_seed=False,
                                 accurate_simulation=self.accurate_monte_carlo,
                                 verbose=False, batch_size=100)

        while p1-p0 > self.percentile_stopping_threshold:
            p = p0 + (p1 - p0)/2
            print(f"Trying p={p} in [{p0},{p1}]")
            perc_predictor = PercentileBasedPredictor(self.base_predictor, percentile=p)

            enforce_deadline_on_partial_solutions = not self.return_frontier
            t0=time.time()
            heft = CloudMOHEFT(self.infra, perc_predictor, self.K, enforce_deadline_on_partial_solutions=enforce_deadline_on_partial_solutions)
            if self.return_frontier:
                sol, _frontier = heft.schedule(job, deadline, return_frontier=True)
                all_solutions.extend(_frontier)
            else:
                sol = heft.schedule(job, deadline, return_frontier=False)
            total_mohe += time.time()-t0

            if sol is None:
                deadline_hits = 0.0
                print(f"Tried p={p} in [{p0},{p1}]: unfeasible")
            else:
                t0=time.time()
                deadline_hits, _, mean_cost,_ = mc.run(sol)
                total_mc += time.time()-t0
                print(f"Tried p={p} in [{p0},{p1}]: {deadline_hits} (desired: {self.required_percentile})")

            if deadline_hits >= self.required_percentile + self.conservative_hit_ratio_margin:
                p1 = p
                heapq.heappush(feasible_solutions, (mean_cost, sol))
            else:
                p0 = p

        if self.return_frontier:
            return (create_frontier(mc, all_solutions, self.required_percentile), mc)

        print(f"MC: {total_mc}")
        print(f"MOHE: {total_mohe}")

        if len(feasible_solutions) > 0:
            return feasible_solutions[0][1] # min cost, feasible solution

        # Use last identified percentile to return a solution (if possible)
        perc_predictor = PercentileBasedPredictor(self.base_predictor, percentile=p1)
        heft = CloudMOHEFT(self.infra, perc_predictor, self.K, enforce_deadline_on_partial_solutions=False)
        sol = heft.schedule(job, deadline)
        if sol is not None:
            setattr(sol, "percentile", p1)
        return sol

def create_frontier (evaluator, solutions, percentile, pool=None):
    # Evaluate all objectives
    objC = {}
    objT = {}
    deadline_feasibility = {}

    if pool is None:
        for i, sol in enumerate(solutions):
            deadline_hits, avg_makespan, avg_cost, _ = evaluator.run(sol)
            objC[sol] = avg_cost
            objT[sol] = avg_makespan
            if deadline_hits >= percentile:
                deadline_feasibility[sol] = 1
            else:
                deadline_feasibility[sol] = 0
    else:
        results = pool.map(evaluator.run, solutions)
        for sol, result, in zip(solutions, results):
            deadline_hits, avg_makespan, avg_cost, _ = result
            objC[sol] = avg_cost
            objT[sol] = avg_makespan
            if deadline_hits >= percentile:
                deadline_feasibility[sol] = 1
            else:
                deadline_feasibility[sol] = 0

    Cmax = max(objC.values())
    Cmin = min(objC.values())
    if Cmax == Cmin:
        Cmax = Cmin + 0.0001
    Tmax = max(objT.values())
    Tmin = min(objT.values())
    if Tmax == Tmin:
        Tmax = Tmin + 0.0001


    # Rank
    Sp = {s: [] for s in solutions}
    np = {s: 0 for s in solutions}
    rank = {}
    fronts = []
    fronts.append([])

    new_solutions = []
    for i,p in enumerate(solutions):
        identical = False
        for j in range(i+1,len(solutions)):
            q = solutions[j]
            if objT[p] == objT[q] and objC[p] == objC[q]:
                identical = True
                break
        if not identical:
            new_solutions.append(p)
    solutions = new_solutions

    for p in solutions:
        for q in solutions:
            if objT[p] < objT[q] and objC[p] < objC[q]:
                Sp[p].append(q)
            elif objT[p] > objT[q] and objC[p] > objC[q]:
                np[p] += 1

        if np[p] == 0:
            rank[p] = 1 # p belongs to Front 1 
            fronts[0].append(p)

    i=0
    while len(fronts[i]) > 0:
        Q = []
        for p in fronts[i]:
            for q in Sp[p]:
                np[q] -= 1

                if np[q] == 0:
                    rank[q] = i+1
                    Q.append(q)
        fronts.append(Q)
        i = i + 1
    fronts = fronts[:-1]

    f = fronts[0]
    frontier = [(objT[sol], objC[sol], deadline_feasibility[sol]) for sol in f]
    
    return frontier

def try_percentile (p, job, deadline, base_predictor, infra, K, return_frontier, mc):
    predictor = PercentileBasedPredictor(base_predictor, percentile=p)
    h = CloudMOHEFT(infra, predictor, K, enforce_deadline_on_partial_solutions=not return_frontier, verbose=False)
    if return_frontier:
        _, frontier = h.schedule(job, deadline, return_frontier)
        return frontier
    else:
        sol = h.schedule(job, deadline, return_frontier)
        if sol is None:
            deadline_hits = 0.0
            mean_cost = 0.0
        else:
            deadline_hits, _, mean_cost,_ = mc.run(sol)
        return (sol, deadline_hits, mean_cost, p)

class ParallelProbMOHEFT (ProbabilisticMOHEFT):

    def __init__ (self, infra, pred, required_percentile=0.9, K=3,
                  return_frontier=False,
                  mc_stopping_rel_error=0.1,
                  conservative_hit_ratio_margin=0.0,
                  accurate_monte_carlo=False,
                  percentiles_count=-1):
        super().__init__(infra,pred,required_percentile,K,return_frontier,
                         mc_stopping_rel_error, conservative_hit_ratio_margin, accurate_monte_carlo)
        self.percentiles_count = percentiles_count
        if self.percentiles_count < 1:
            self.percentiles_count = os.cpu_count()


    def schedule (self, job, deadline):
        p0 = 0.1 
        p1 = 0.99

        feasible_min_cost = None
        feasible_min_cost_sol = None
        unfeasible_max_hits = None
        unfeasible_max_hits_sol = None

        enforce_deadline_on_partial_solutions = not self.return_frontier

        mc = MonteCarloEvaluator(job, self.base_predictor, deadline,
                                 self.required_percentile, 
                                 max_relative_error=self.mc_stopping_rel_error,
                                 change_seed=False,
                                 accurate_simulation=self.accurate_monte_carlo,
                                 verbose=False, batch_size=100)

        percentiles_to_try = np.linspace(p0, p1, self.percentiles_count)

        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(functools.partial(try_percentile,
                                                 base_predictor=self.base_predictor,
                                                 K=self.K,
                                                 deadline=deadline,
                                                 job=job,
                                                 mc=mc,
                                                 return_frontier=self.return_frontier,
                                                 infra=self.infra), percentiles_to_try)
            if self.return_frontier:
                all_solutions = [] # for frontier
                for frontier in results:
                    all_solutions.extend(frontier)
                    return (create_frontier(mc, all_solutions, self.required_percentile, pool=pool), mc)
            else:
                for sol,deadline_hits,mean_cost,_ in results:
                    if deadline_hits >= self.required_percentile + self.conservative_hit_ratio_margin:
                        if feasible_min_cost is None or mean_cost < feasible_min_cost:
                            feasible_min_cost = mean_cost
                            feasible_min_cost_sol = sol
                    else:
                        if unfeasible_max_hits is None or deadline_hits > unfeasible_max_hits:
                            unfeasible_max_hits = deadline_hits
                            unfeasible_max_hits_sol = sol

        if feasible_min_cost_sol is not None:
            return feasible_min_cost_sol
        else:
            return unfeasible_max_hits_sol

class ParallelProbMOHEFT2 (ProbabilisticMOHEFT):

    def __init__ (self, infra, pred, required_percentile=0.9, K=3,
                  return_frontier=False,
                  mc_stopping_rel_error=0.1,
                  conservative_hit_ratio_margin=0.0,
                  accurate_monte_carlo=False,
                  percentiles_count=-1):
        super().__init__(infra,pred,required_percentile,K,return_frontier,
                         mc_stopping_rel_error, conservative_hit_ratio_margin, accurate_monte_carlo)
        self.percentiles_count = percentiles_count
        if self.percentiles_count < 1:
            self.percentiles_count = os.cpu_count()


    def schedule (self, job, deadline):
        feasible_min_cost = None
        feasible_min_cost_sol = None
        unfeasible_max_hits = None
        unfeasible_max_hits_sol = None
        best_p = None
        best_unf_p = None

        enforce_deadline_on_partial_solutions = not self.return_frontier

        mc = MonteCarloEvaluator(job, self.base_predictor, deadline,
                                 self.required_percentile, 
                                 max_relative_error=self.mc_stopping_rel_error,
                                 change_seed=False,
                                 accurate_simulation=self.accurate_monte_carlo,
                                 verbose=False, batch_size=100)
        
        interval_width = 1.0/self.percentiles_count
        percentiles_to_try = np.linspace(interval_width/2, 1.0-interval_width/2, self.percentiles_count)
        print(f"Trying: {percentiles_to_try}")

        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(functools.partial(try_percentile,
                                                 base_predictor=self.base_predictor,
                                                 K=self.K,
                                                 deadline=deadline,
                                                 job=job,
                                                 mc=mc,
                                                 return_frontier=self.return_frontier,
                                                 infra=self.infra), percentiles_to_try)
            if self.return_frontier:
                all_solutions = [] # for frontier
                for frontier in results:
                    all_solutions.extend(frontier)
                    return (create_frontier(mc, all_solutions, self.required_percentile, pool=pool), mc)
            else:
                for sol,deadline_hits,mean_cost,p in results:
                    if deadline_hits >= self.required_percentile + self.conservative_hit_ratio_margin:
                        if feasible_min_cost is None or mean_cost < feasible_min_cost:
                            feasible_min_cost = mean_cost
                            feasible_min_cost_sol = sol
                            best_p = p
                    else:
                        if unfeasible_max_hits is None or deadline_hits > unfeasible_max_hits:
                            unfeasible_max_hits = deadline_hits
                            unfeasible_max_hits_sol = sol
                            best_unf_p = p

        # Step 2: partition around the best percentile
        if feasible_min_cost_sol is not None:
            p = best_p
        else:
            p = best_unf_p

        percentiles_to_try = np.linspace(max(0.01, p-interval_width/2), min(0.99,p+interval_width/2), self.percentiles_count)
        print(f"Trying: {percentiles_to_try}")

        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(functools.partial(try_percentile,
                                                 base_predictor=self.base_predictor,
                                                 K=self.K,
                                                 deadline=deadline,
                                                 job=job,
                                                 mc=mc,
                                                 return_frontier=self.return_frontier,
                                                 infra=self.infra), percentiles_to_try)
            for sol,deadline_hits,mean_cost,p in results:
                if deadline_hits >= self.required_percentile + self.conservative_hit_ratio_margin:
                    if feasible_min_cost is None or mean_cost < feasible_min_cost:
                        feasible_min_cost = mean_cost
                        feasible_min_cost_sol = sol
                        best_p = p
                else:
                    if unfeasible_max_hits is None or deadline_hits > unfeasible_max_hits:
                        unfeasible_max_hits = deadline_hits
                        unfeasible_max_hits_sol = sol
                        best_unf_p = p


        if feasible_min_cost_sol is not None:
            return feasible_min_cost_sol
        else:
            return unfeasible_max_hits_sol

class PercentileBasedPredictor:

    def __init__ (self, base_predictor, percentile=0.9):
        self.base = base_predictor
        self.percentile = percentile

    def exec_time (self, op: Operator, job: Job, vm_type, first_on_the_machine=False, first_in_the_graph=False):
        distribution = self.base.get_exec_time_distribution(op, job, vm_type, first_on_the_machine, first_in_the_graph)
        assert(distribution is not None)

        # return percentile
        return distribution.get_percentile(self.percentile)

    def data_writing_time (self, op, vm_type=None):
        return self.base.data_writing_time(op, vm_type)

    def data_reading_time (self, op1, op2, vm_type=None):
        return self.base.data_reading_time(op1, op2, vm_type)

