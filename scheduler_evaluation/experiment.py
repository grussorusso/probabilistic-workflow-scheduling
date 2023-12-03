import sys
import networkx as nx
import numpy as np
import time

from scheduler.infrastructure import *
from scheduler.scheduling import *
from scheduler.algorithms import Scheduler
from scheduler.job import *
from scheduler import evaluation


def run(infra, job, predictor, args, detailed_results=False):
    scheduler = Scheduler(infra)

    # ---------------------------------------------------------------------------
    N=args.ntasks
    assert(N >= 1)
    deadline = args.deadline
    assert(deadline > 0.0)

    multitask_job = job.to_multidataset_job(N)

    t0 = time.time()
    sol = scheduler.schedule(multitask_job, predictor, deadline,
            percentile=args.deadline_percentile, algorithm=args.algorithm,
                             other_opts=args)
    sched_time = time.time()-t0


    if sol is None:
        print("Scheduling failed.")
    else:
        print(sol)

    mc = evaluation.MonteCarloEvaluator(multitask_job, predictor,
                                deadline, args.deadline_percentile, 
                                max_evaluations=args.evaluations,
                                max_relative_error=0.01,
                                accurate_simulation=True,
                                billing_period=args.billing_period_sec,
                                change_seed=False, verbose=True, batch_size=100)

    results = mc.run(sol, initial_seed=args.evaluation_seed, detailed_results=detailed_results)

    if detailed_results:
        print(f"Deadline satisfaction: {results.hit_ratio*100} %")
        print(f"Average cost: {results.avg_cost}")
        print(f"Average makespan: {results.avg_makespan}")
        print(f"Average tardiness: {results.avg_tardiness}")
        print(f"Makespan percentiles: {results.makespan_quantiles}")
        print(f"MC runs: {results.total_runs}")
        print(f"Unfeasible runs: {results.unfeasible_runs}")
    return sol, results, sched_time

