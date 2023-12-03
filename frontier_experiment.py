import sys
import re
import time
import os
import argparse
import networkx as nx
import numpy as np
import pandas as pd

from scheduler.infrastructure import *
from scheduler.job import *
from scheduler.provider import FakeProvider, FakeVMType
from scheduler.prediction import SimplePredictor, UniversalScalabilityFunction
from scheduler.probabilistic import ProbabilisticMOHEFT, ParallelProbMOHEFT, create_frontier
from scheduler.genetic import GeneticScheduler
from scheduler import algorithms
from scheduler import distributions
from scheduler_evaluation import experiment, jobs

FAMILY_SPEEDUP = {"c4": 0.8, "c5": 1.0, "m5": 0.8}
MACHINES_CSV="data/machines.csv"

def create_job (name):
    if name == "montage":
        return jobs.create_montage()
    elif name == "cybershake":
        return jobs.create_cybershake()
    elif name == "sipht":
        return jobs.create_sipht()
    elif "sipht" in name:
        m = re.match("sipht(\d+)", name)
        n = int(m.groups()[0])
        return jobs.create_sipht(n)
    elif name == "ligo":
        return jobs.create_ligo()
    elif name == "epigenomics":
        return jobs.create_epigenomics(4)
    elif "epigenomics" in name:
        m = re.match("epigenomics(\d+)", name)
        n = int(m.groups()[0])
        return jobs.create_epigenomics(n)
    else:
        raise RuntimeError("Unknown job name: " + str(name))


scalability_fun = UniversalScalabilityFunction(0.01, 0.0, single_core_speedup=0.1)

def create_gamma_distributions (mean_times, scv=1.0):
    op_distributions = {}
    for op,mean in mean_times.items():
        op_distributions[op] = distributions.Gamma(mean, scv)
    return op_distributions

def create_deterministic_distributions (mean_times):
    op_distributions = {}
    for op,mean in mean_times.items():
        op_distributions[op] = distributions.Deterministic(mean)
    return op_distributions

def create_uniform_distributions (mean_times, scv=0.5):
    op_distributions = {}
    for op,mean in mean_times.items():
        op_distributions[op] = distributions.Uniform(mean, scv)
    return op_distributions

def create_halfnormal_distributions (mean_times):
    op_distributions = {}
    for op,mean in mean_times.items():
        op_distributions[op] = distributions.HalfNormal(mean)
    return op_distributions

def print_results (results, filename=None):
    for line in results:
        print(line)
    if filename is not None:
        with open(filename, "w") as of:
            for line in results:
                print(line,file=of)

def create_provider_infrastructure (max_vmtypes=-1, random_vm_selection=False, provider_capacity=-1, skip_first_machines=0):
    provider = FakeProvider(None)
    if provider_capacity > 0:
        provider.set_vcpu_limit(provider_capacity)
    vm_types = load_vmtypes_from_csv(MACHINES_CSV, provider, max_vmtypes, random_vm_selection, skip_first_machines=skip_first_machines)
    return Infrastructure(vm_types)

def load_vmtypes_from_csv (csvfile, provider, max_vmtypes=-1, random_selection=False, skip_first_machines=0):
    vmtypes = []
    with open(csvfile, "r") as f:
        # skip headers
        f.readline()
        for i in range(skip_first_machines):
            f.readline()
        for line in f:
            fields = line.strip().split(",")
            cost, core_count, name, memory_size, family, bandwidth_MBs = fields
            cost = float(cost) / 3600.0 # hourly to per-second cost
            vmt = FakeVMType(name, family, provider, int(core_count),
                             0, float(memory_size), "", cost, float(bandwidth_MBs))
            vmtypes.append(vmt)

    if max_vmtypes > 0:
        if random_selection:
            rng = np.random.default_rng(seed=1)
            return rng.choice(vmtypes, size=(max_vmtypes,))
        else:
            return vmtypes[:max_vmtypes]

    return vmtypes



def main(args):
    aws = FakeProvider(None) 
    aws.set_vcpu_limit(args.max_vcpu)


    vm_types = load_vmtypes_from_csv(MACHINES_CSV, aws, args.max_vmtypes)
    print(f"VM_Types: {len(vm_types)}")
    print(f"Deadline: {args.deadline}")

    infra = Infrastructure(vm_types)

    # ---------------------------------------------------------------------------
    job, mean_exec_times, op_output_mb = create_job(args.job)

    # Set operator distributions
    op_distributions = create_gamma_distributions(mean_exec_times)
    predictor = SimplePredictor(op_distributions, scalability_fun, op_output_mb={})

    multitask_job = job.to_multidataset_job(1)

    t0 = time.time()
    #ph = ProbabilisticMOHEFT(infra, predictor, K=args.moheft_k,
    #                            required_percentile=args.deadline_percentile,
    #                            percentile_stopping_threshold=args.percentile_epsilon,
    #                            return_frontier=True,
    #                                mc_stopping_rel_error = args.mc_stopping_rel_error)
    ph = ParallelProbMOHEFT(infra, predictor, K=args.moheft_k,
                                required_percentile=args.deadline_percentile,
                                percentiles_count=max(16,os.cpu_count()),
                                return_frontier=True,
                                    mc_stopping_rel_error = args.mc_stopping_rel_error)
    frontier, mc_evaluator = ph.schedule(multitask_job, args.deadline)
    sched_time = time.time()-t0

    print(sched_time)

    frontier.sort()

    with open("frontier.txt", "w") as of:
        print(f"#SchedTime: {sched_time}", file=of)
        for x,y,d in frontier:
            print(f"{x}\t{y}\t{d}", file=of)

    COMPARE_GENETIC=True
    if COMPARE_GENETIC:
        for iters in [50000,100000]:
            s = GeneticScheduler(infra, predictor, iters, deadline_percentile=args.deadline_percentile,
                                            accurate_monte_carlo = False,
                                                return_frontier=True,
                                            mc_stopping_rel_error = args.mc_stopping_rel_error)
            genetic_frontier = s.schedule(multitask_job, args.deadline)
            sched_time = time.time()-t0
            print(sched_time)

            frontier2 = create_frontier(mc_evaluator, genetic_frontier, args.deadline_percentile)

            frontier2.sort()

            with open(f"frontierGenetic-{iters}.txt", "w") as of:
                print(f"#SchedTime: {sched_time}", file=of)
                for x,y,d in frontier2:
                    print(f"{x}\t{y}\t{d}", file=of)


    import matplotlib.pyplot as plt
    X = [f[0] for f in frontier]
    Y = [f[1] for f in frontier]
    plt.plot(X, Y, '--', label="ProbMOHEFT")

    X = [f[0] for f in frontier if f[2] == 0]
    Y = [f[1] for f in frontier if f[2] == 0]
    plt.scatter(X, Y, c="red")

    X = [f[0] for f in frontier if f[2] == 1]
    Y = [f[1] for f in frontier if f[2] == 1]
    plt.scatter(X, Y, c="green")

    if COMPARE_GENETIC:
        X = [f[0] for f in frontier2]
        Y = [f[1] for f in frontier2]
        plt.plot(X, Y, '--', c="magenta", label="Genetic")

        X = [f[0] for f in frontier2 if f[2] == 0]
        Y = [f[1] for f in frontier2 if f[2] == 0]
        plt.scatter(X, Y, c="red")

        X = [f[0] for f in frontier2 if f[2] == 1]
        Y = [f[1] for f in frontier2 if f[2] == 1]
        plt.scatter(X, Y, c="green")

    plt.ylabel('Avg Cost')
    plt.xlabel('Avg Makespan')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', action='store', required=False, default="epigenomics", type=str)
    parser.add_argument('--max_vmtypes', action='store', required=False, default=8, type=int)
    parser.add_argument('--max_vcpu', action='store', required=False, default=576, type=int)
    parser.add_argument('--moheft_k', action='store', required=False, default=50, type=int)
    parser.add_argument('--deadline', action='store', required=False, default=3600.0, type=float)
    parser.add_argument('--deadline_percentile', action='store', required=False, default=0.9, type=float)
    parser.add_argument('--evaluation_seed', action='store', required=False, default=4123, type=int)
    parser.add_argument('--mc_stopping_rel_error', action='store', required=False, default=0.05, type=float)
    parser.add_argument('--percentile_epsilon', action='store', required=False, default=0.02, type=float)

    args = parser.parse_args()

    main(args)
