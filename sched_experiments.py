import sys
import re
import argparse
import networkx as nx
import numpy as np
import pandas as pd

from xClouder.scheduler.infrastructure import *
from xClouder.scheduler.job import *
from xClouder.scheduler.gpf import GPFGraph
from xClouder.scheduler.provider import FakeProvider, FakeVMType
from xClouder.scheduler.prediction import SimplePredictor, UniversalScalabilityFunction
from xClouder.scheduler import algorithms
from xClouder.scheduler import distributions
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
    elif name == "dummy-preprocessing":
        return jobs.create_dummy_pipeline(9)
    elif name == "dummy-ndvi":
        return jobs.create_dummy_pipeline(5)
    elif name == "dummy-firedetection":
        return jobs.create_dummy_fire_detection()
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

def create_uniform_distributions (mean_times):
    op_distributions = {}
    for op,mean in mean_times.items():
        op_distributions[op] = distributions.Uniform(mean)
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

    #scalability_fun = lambda cores : cores*0.5+0.5

    vm_types = load_vmtypes_from_csv(MACHINES_CSV, aws, args.max_vmtypes)
    print(f"VM_Types: {len(vm_types)}")
    print(f"Deadline: {args.deadline}")

    infra = Infrastructure(vm_types)

    # ---------------------------------------------------------------------------
    if args.job_as_xmlgraph is None:
        job, mean_exec_times, op_output_mb = create_job(args.job)
    else:
        print("Reading GPFGraph from {job_name}...")
        job = GPFGraph(xml_path=args.job_as_xmlgraph)
        mean_exec_times = None

    if mean_exec_times is None:
        print("Generating random execution times...")
        exec_time_rng = np.random.default_rng(seed=args.durations_seed)
        mean_exec_times = {op: exec_time_rng.uniform(4.0, 300.0) for op in job.nodes()}
    # Set operator distributions
    op_distributions = create_gamma_distributions(mean_exec_times, 1.0)
    predictor = SimplePredictor(op_distributions, scalability_fun, op_output_mb=op_output_mb)

    _, results, sched_time = experiment.run(infra, job, predictor, args, detailed_results=True)
    print(results)
    print(sched_time)

def experiment_capacity (args):
    results = []
    args.genetic_max_evaluations = 50000
    args.deadline_percentile = 0.9
    args.mc_stopping_rel_error = 0.05
    args.billing_period_sec = 1
    args.moheft_k = 10

    JOBS = ["epigenomics", "cybershake", "montage", "sipht", "ligo"]
    DEADLINES = [900, 900, 900, 2400, 1800]

    outfile="resultsCapacity.csv"
    # Check existing results
    try:
        old_results = pd.read_csv(outfile)
        if not "AccurateMC" in old_results.columns:
            old_results.loc[:,"AccurateMC"] = False
    except:
        old_results = None

    for deadline,job_name in zip(DEADLINES, JOBS): 
        job, mean_exec_times, op_output_mb = create_job(job_name)
        op_distributions = create_gamma_distributions(mean_exec_times)
        predictor = SimplePredictor(op_distributions, scalability_fun, FAMILY_SPEEDUP, op_output_mb=op_output_mb)
        for deadline_coeff in [1.5,2.0,3.0]:
            args.deadline = deadline*deadline_coeff
            for n_vmtypes in [8]:
                skip_vmtypes = 3 if n_vmtypes < 8 else 0
                for accurate_mc in [True, False]:
                    args.accurate_mc = accurate_mc
                    for alg in [algorithms.SCHED_HEFT, algorithms.SCHED_ProbMOHEFT, algorithms.SCHED_GENETIC]:
                        args.algorithm = alg
                        for capacity in [25, 50, 100, 400]:
                            infra = create_provider_infrastructure(n_vmtypes, provider_capacity=capacity, skip_first_machines=skip_vmtypes)

                            # Check if we can skip this run
                            if old_results is not None:
                                if not old_results[(old_results.Algorithm == alg) &\
                                        (old_results.Job == job_name) &\
                                        (old_results.Capacity == capacity) &\
                                        (old_results.Deadline == args.deadline) &\
                                        (old_results.AccurateMC == accurate_mc) &\
                                        (old_results.VMTypes == n_vmtypes)].empty:
                                    print("Skipping conf")
                                    continue
                            if (alg != algorithms.SCHED_ProbMOHEFT and accurate_mc):
                                continue

                            sol, eval_results, sched_time = experiment.run(infra, job, predictor, args, detailed_results=True)

                            result = {}
                            result["VMTypes"] = n_vmtypes
                            result["Job"] = job_name
                            result["Deadline"] = args.deadline
                            result["Percentile"] = args.deadline_percentile
                            result["Algorithm"] = alg
                            result["Capacity"] = capacity
                            result["AccurateMC"] = accurate_mc
                            result["AvgCost"] = eval_results.avg_cost
                            result["AvgMakespan"] = eval_results.avg_makespan
                            result["StdCost"] = eval_results.std_cost
                            result["StdMakespan"] = eval_results.std_makespan
                            result["HitRatio"] = eval_results.hit_ratio
                            result["AvgTardiness"] = eval_results.avg_tardiness
                            result["MCRuns"] = eval_results.total_runs
                            result["MCUnfeasibleRuns"] = eval_results.unfeasible_runs
                            for ip,p in enumerate(eval_results.percentiles):
                                result[f"Makespan-P{p}"] = eval_results.makespan_quantiles[ip]
                            for ip,p in enumerate(eval_results.percentiles):
                                result[f"Cost-P{p}"] = eval_results.cost_quantiles[ip]
                            result["SchedulingTime"] = sched_time

                            results.append(result)
                            print(result)
                            # save partial results
                            resultsDf = pd.DataFrame(results)
                            if old_results is not None:
                                resultsDf = pd.concat([old_results, resultsDf])
                            resultsDf.to_csv(outfile, index=False)

    resultsDf = pd.DataFrame(results)
    if old_results is not None:
        resultsDf = pd.concat([old_results, resultsDf])
    resultsDf.to_csv(outfile, index=False)

def experiment_genetic2(args):
    n_vmtypes=4
    infra = create_provider_infrastructure(n_vmtypes, skip_first_machines=3)

    results = []
    args.deadline = 600
    args.deadline_percentile = 0.9
    args.algorithm = algorithms.SCHED_GENETIC

    outfile="resultsGenetic2.csv"

    for job_name in ["epigenomics"]:
        job, mean_exec_times, op_output_mb = create_job(job_name)
        op_distributions = create_gamma_distributions(mean_exec_times, 1.0)
        predictor = SimplePredictor(op_distributions, scalability_fun, FAMILY_SPEEDUP, op_output_mb=op_output_mb,
                                    vm_startup_time=0)
        for seed in [5, 10, 1222, 150, 201, 8292, 12763, 18182]:
            args.scheduler_seed = seed
            for rel_error in [0.02, 0.025, 0.1, 0.05]:
                args.mc_stopping_rel_error = rel_error
                for accurate_mc in [False]:
                    args.accurate_mc = accurate_mc
                    for iterations in [50000]:
                        args.genetic_max_evaluations = iterations
                        sol, eval_results, sched_time = experiment.run(infra, job, predictor, args)

                        avg_cost = eval_results[1] if eval_results is not None else -1.0
                        satisfaction = eval_results[0] if eval_results is not None else -1.0
                        makespan_quantile = eval_results[2] if eval_results is not None else 0.0

                        result = {}
                        result["Job"] = job_name
                        result["Deadline"] = args.deadline
                        result["Accurate"] = args.accurate_mc
                        result["RelError"] = args.mc_stopping_rel_error
                        result["Evaluations"] = args.genetic_max_evaluations
                        result["Seed"] = seed
                        result["AvgCost"] = avg_cost
                        result["Satisfaction"] = satisfaction
                        result["MakespanQuantile"] = makespan_quantile
                        result["SchedulingTime"] = sched_time

                        results.append(result)
                        print(result)
                        # save partial results
                        df = pd.DataFrame(results)
                        df.to_csv(outfile, index=False)

    df = pd.DataFrame(results)
    df.to_csv(outfile, index=False)
def experiment_genetic(args):
    n_vmtypes=4
    infra = create_provider_infrastructure(n_vmtypes, skip_first_machines=3)

    results = []
    args.deadline = 600
    args.deadline_percentile = 0.9
    args.algorithm = algorithms.SCHED_GENETIC

    outfile="resultsGenetic.csv"

    for job_name in ["epigenomics"]:
        job, mean_exec_times, op_output_mb = create_job(job_name)
        op_distributions = create_gamma_distributions(mean_exec_times, 1.0)
        predictor = SimplePredictor(op_distributions, scalability_fun, FAMILY_SPEEDUP, op_output_mb=op_output_mb,
                                    vm_startup_time=0)
        for initial_population in [1000, 5000, 100]:
            args.genetic_population = initial_population
            for seed in [5, 10, 1222, 150, 201, 8292, 12763, 18182]:
                args.scheduler_seed = seed
                for iterations in [50000, 100000, 500000]:
                    args.genetic_max_evaluations = iterations
                    sol, eval_results, sched_time = experiment.run(infra, job, predictor, args)

                    avg_cost = eval_results[1] if eval_results is not None else -1.0
                    satisfaction = eval_results[0] if eval_results is not None else -1.0
                    makespan_quantile = eval_results[2] if eval_results is not None else 0.0

                    result = {}
                    result["Job"] = job_name
                    result["Deadline"] = args.deadline
                    result["Population"] = args.genetic_population
                    result["Evaluations"] = args.genetic_max_evaluations
                    result["Seed"] = seed
                    result["AvgCost"] = avg_cost
                    result["Satisfaction"] = satisfaction
                    result["MakespanQuantile"] = makespan_quantile
                    result["SchedulingTime"] = sched_time

                    results.append(result)
                    print(result)
                    # save partial results
                    df = pd.DataFrame(results)
                    df.to_csv(outfile, index=False)

    df = pd.DataFrame(results)
    df.to_csv(outfile, index=False)

def experiment_probmoheft(args):
    infra = create_provider_infrastructure(8) # single family

    results = []
    args.algorithm = algorithms.SCHED_ProbMOHEFT

    outfile="resultsProbMOHEFT.csv"

    JOBS = ["epigenomics", "cybershake", "montage", "sipht", "ligo"]
    DEADLINES = [900, 900, 900, 2400, 1800]
    CONSERVATIVE_MARGINS=[0.0, 0.01, 0.025]

    for job_name,deadline in zip(JOBS, DEADLINES):
        args.deadline = deadline
        job, mean_exec_times, op_output_mb = create_job(job_name)
        op_distributions = create_gamma_distributions(mean_exec_times, 1.0)
        predictor = SimplePredictor(op_distributions, scalability_fun,
                                    FAMILY_SPEEDUP, op_output_mb=op_output_mb)

        for accurate_mc in [False,True]:
            args.accurate_mc = accurate_mc
            for epsilon in [0.1, 0.05, 0.025, 0.02]:
                for K in [5, 10, 15, 20]:
                    args.moheft_k = K
                    args.percentile_epsilon = epsilon
                    for stopping_error in [0.05]:
                    #for stopping_error in [0.05, 0.025]:
                        args.mc_stopping_rel_error = stopping_error
                        for percentile in [0.9, 0.95]:
                            args.deadline_percentile = percentile
                            for percentile_margin in [0, 0.01, 0.025, 0.05]:
                                args.percentile_margin = percentile_margin
                                sol, eval_results, sched_time = experiment.run(infra, job, predictor, args)

                                satisfaction = eval_results[0] if eval_results is not None else -1.0
                                avg_cost = eval_results[2] if eval_results is not None else -1.0
                                makespan_quantile = eval_results[3] if eval_results is not None else 0.0

                                result = {}
                                result["Job"] = job_name
                                result["Deadline"] = args.deadline
                                result["Percentile"] = percentile
                                result["K"] = K
                                result["Epsilon"] = epsilon
                                result["AccurateMC"] = accurate_mc
                                result["StoppingErorr"] = stopping_error
                                result["PercentileMargin"] = percentile_margin
                                result["AvgCost"] = avg_cost
                                result["Satisfaction"] = satisfaction
                                result["MakespanQuantile"] = makespan_quantile
                                result["SchedulingTime"] = sched_time

                                results.append(result)
                                print(result)
                                df = pd.DataFrame(results)
                                df.to_csv(outfile, index=False)

    df = pd.DataFrame(results)
    df.to_csv(outfile, index=False)

def experiment_distributions(args):
    results = []
    outfile="resultsDistributions.csv"

    JOBS = ["epigenomics", "cybershake", "montage", "sipht", "ligo"]
    DEADLINES = [900, 900, 900, 2400, 1800]
    SEEDS = [1]

    args.genetic_max_evaluations = 50000
    args.mc_stopping_rel_error = 0.05
    args.billing_period_sec = 1
    args.moheft_k = 10
    args.percentiles_to_try = 8
    args.deadline_percentile = 0.9

    my_distributions = ["Uniform", "Deterministic", "HalfNormal"]

    single_core_speedup=0.1
    SCALABILITY_FUNCTIONS = [UniversalScalabilityFunction(0.01, 0.0, single_core_speedup=single_core_speedup,name="a10-2"),
            UniversalScalabilityFunction(0.0, 0.0, single_core_speedup=single_core_speedup,name="Linear"),
            UniversalScalabilityFunction(0.0001, 0.001, single_core_speedup=single_core_speedup,name="a10-4_b10-3")]

    # Check existing results
    try:
        old_results = pd.read_csv(outfile)
    except:
        old_results = None

    for seed in SEEDS:
        args.scheduler_seed = seed
        for alg in [algorithms.SCHED_ProbMOHEFT, algorithms.SCHED_CloudMOHEFT, algorithms.SCHED_HEFT, algorithms.SCHED_GENETIC]:
            args.algorithm = alg
            for scal_fun in SCALABILITY_FUNCTIONS:
                for n_vmtypes in [8]:
                    infra = create_provider_infrastructure(n_vmtypes)
                    for job_name, deadline in zip(JOBS, DEADLINES):
                        job, mean_exec_times, op_output_mb = create_job(job_name)
                        args.deadline = deadline
                        for d in my_distributions:
                            if d == "Gamma":
                                op_distributions = create_gamma_distributions(mean_exec_times, 1.0)
                            if d == "Deterministic":
                                op_distributions = create_deterministic_distributions(mean_exec_times)
                            elif d == "Uniform":
                                op_distributions = create_uniform_distributions(mean_exec_times)
                            elif d == "HalfNormal":
                                op_distributions = create_halfnormal_distributions(mean_exec_times)

                            predictor = SimplePredictor(op_distributions, scal_fun, FAMILY_SPEEDUP, op_output_mb=op_output_mb)

                            for percentile in [0.9]:
                                args.deadline_percentile = percentile

                                # Check if we can skip this run
                                if old_results is not None:
                                    if not old_results[(old_results.Algorithm == alg) &\
                                            (old_results.Job == job_name) &\
                                            (old_results.ScalFunction == scal_fun.name) &\
                                            (old_results.Seed == seed) &\
                                            (old_results.Deadline == args.deadline) &\
                                            (old_results.Distribution == d) &\
                                            (old_results.VMTypes == n_vmtypes) &\
                                            (old_results.Percentile == percentile)].empty:
                                        print("Skipping conf")
                                        continue
                                if alg != "Genetic" and seed != SEEDS[0]:
                                    continue # we only change seed for Genetic

                                sol, eval_results, sched_time = experiment.run(infra, job, predictor, args, detailed_results=True)

                                result = {}
                                result["VMTypes"] = n_vmtypes
                                result["Job"] = job_name
                                result["Nodes"] = len(list(job.nodes()))
                                result["Deadline"] = args.deadline
                                result["Percentile"] = percentile
                                result["Algorithm"] = alg
                                result["Distribution"] = d
                                result["ScalFunction"] = scal_fun.name
                                result["Seed"] = seed
                                result["AvgCost"] = eval_results.avg_cost
                                result["AvgMakespan"] = eval_results.avg_makespan
                                result["StdCost"] = eval_results.std_cost
                                result["StdMakespan"] = eval_results.std_makespan
                                result["HitRatio"] = eval_results.hit_ratio
                                result["AvgTardiness"] = eval_results.avg_tardiness
                                result["MCRuns"] = eval_results.total_runs
                                result["MCUnfeasibleRuns"] = eval_results.unfeasible_runs
                                for ip,p in enumerate(eval_results.percentiles):
                                    result[f"Makespan-P{p}"] = eval_results.makespan_quantiles[ip]
                                for ip,p in enumerate(eval_results.percentiles):
                                    result[f"Cost-P{p}"] = eval_results.cost_quantiles[ip]
                                result["SchedulingTime"] = sched_time

                                results.append(result)
                                print(result)
                                # save partial results
                                resultsDf = pd.DataFrame(results)
                                if old_results is not None:
                                    resultsDf = pd.concat([old_results, resultsDf])
                                resultsDf.to_csv(outfile, index=False)

    resultsDf = pd.DataFrame(results)
    if old_results is not None:
        resultsDf = pd.concat([old_results, resultsDf])
    resultsDf.to_csv(outfile, index=False)

def experiment_scalability2(args):
    results = []
    args.mc_stopping_rel_error = 0.05
    args.billing_period_sec = 1
    args.moheft_k = 10
    args.percentiles_to_try = 8
    args.deadline_percentile = 0.9
    outfile="resultsScalability.csv"

    JOBS = ["epigenomics"]
    BASE_DEADLINES = [1500]
    assert(len(BASE_DEADLINES)==len(JOBS))

    # Check existing results
    try:
        old_results = pd.read_csv(outfile)
    except:
        old_results = None

    
    for alg in [algorithms.SCHED_ParallelProbMOHEFT, algorithms.SCHED_ParallelProbMOHEFT2]:
        args.algorithm = alg
        for n_vmtypes in [8]:
            infra = create_provider_infrastructure(n_vmtypes)
            for base_job_name, base_deadline in zip(JOBS, BASE_DEADLINES):
                for ijs,jobScale in enumerate([4]):
                    job_name = f"{base_job_name}{jobScale}"
                    job, mean_exec_times, op_output_mb = create_job(job_name)
                    args.deadline = base_deadline * 1.1**ijs
                    op_distributions = create_gamma_distributions(mean_exec_times, 1.0)
                    predictor = SimplePredictor(op_distributions, scalability_fun, FAMILY_SPEEDUP, op_output_mb=op_output_mb)

                    # Check if we can skip this run
                    if old_results is not None:
                        if not old_results[(old_results.Algorithm == alg) &\
                                (old_results.Job == base_job_name) &\
                                (old_results.Nodes == len(list(job.nodes()))) &\
                                (old_results.VMTypes == n_vmtypes) &\
                                (old_results.Percentile == args.deadline_percentile)].empty:
                            print("Skipping conf")
                            continue
                    if jobScale > 256 and base_job_name == "epigenomics":
                        continue

                    sol, eval_results, sched_time = experiment.run(infra, job, predictor, args, detailed_results=True)

                    result = {}
                    result["VMTypes"] = n_vmtypes
                    result["Job"] = base_job_name
                    result["Nodes"] = len(list(job.nodes()))
                    result["Deadline"] = args.deadline
                    result["Percentile"] = args.deadline_percentile
                    result["Algorithm"] = alg
                    result["SchedulingTime"] = sched_time
                    result["AvgCost"] = eval_results.avg_cost
                    result["AvgMakespan"] = eval_results.avg_makespan
                    result["StdCost"] = eval_results.std_cost
                    result["StdMakespan"] = eval_results.std_makespan
                    result["HitRatio"] = eval_results.hit_ratio
                    result["AvgTardiness"] = eval_results.avg_tardiness
                    result["MCRuns"] = eval_results.total_runs
                    result["MCUnfeasibleRuns"] = eval_results.unfeasible_runs
                    for ip,p in enumerate(eval_results.percentiles):
                        result[f"Makespan-P{p}"] = eval_results.makespan_quantiles[ip]
                    for ip,p in enumerate(eval_results.percentiles):
                        result[f"Cost-P{p}"] = eval_results.cost_quantiles[ip]

                    results.append(result)
                    print(result)
                    # save partial results
                    resultsDf = pd.DataFrame(results)
                    if old_results is not None:
                        resultsDf = pd.concat([old_results, resultsDf])
                    resultsDf.to_csv(outfile, index=False)
    
    resultsDf = pd.DataFrame(results)
    if old_results is not None:
        resultsDf = pd.concat([old_results, resultsDf])
    resultsDf.to_csv(outfile, index=False)
def experiment_scalability(args):
    results = []
    args.mc_stopping_rel_error = 0.05
    args.billing_period_sec = 1
    args.moheft_k = 10
    args.percentiles_to_try = 8
    args.deadline_percentile = 0.9
    outfile="resultsScalability.csv"

    JOBS = ["epigenomics", "sipht",]
    BASE_DEADLINES = [900, 2400]
    assert(len(BASE_DEADLINES)==len(JOBS))

    # Check existing results
    try:
        old_results = pd.read_csv(outfile)
    except:
        old_results = None

    
    for n_vmtypes in [8,13,21]:
        for alg in [algorithms.SCHED_ParallelProbMOHEFT, algorithms.SCHED_ParallelProbMOHEFT2, algorithms.SCHED_ProbMOHEFT]:
            args.algorithm = alg
            for base_job_name, base_deadline in zip(JOBS, BASE_DEADLINES):
                for ijs,jobScale in enumerate([1,2,4,8,16,32,64,128,256,512,1024]):
                    job_name = f"{base_job_name}{jobScale}"
                    job, mean_exec_times, op_output_mb = create_job(job_name)
                    args.deadline = base_deadline * 1.1**ijs
                    op_distributions = create_gamma_distributions(mean_exec_times, 1.0)
                    predictor = SimplePredictor(op_distributions, scalability_fun, FAMILY_SPEEDUP, op_output_mb=op_output_mb)

                    # Check if we can skip this run
                    if old_results is not None:
                        if not old_results[(old_results.Algorithm == alg) &\
                                (old_results.Job == base_job_name) &\
                                (old_results.Nodes == len(list(job.nodes()))) &\
                                (old_results.VMTypes == n_vmtypes) &\
                                (old_results.Percentile == args.deadline_percentile)].empty:
                            print("Skipping conf")
                            continue
                    if jobScale > 256 and base_job_name == "epigenomics":
                        continue

                    infra = create_provider_infrastructure(n_vmtypes)
                    sol, eval_results, sched_time = experiment.run(infra, job, predictor, args, detailed_results=True)

                    result = {}
                    result["VMTypes"] = n_vmtypes
                    result["Job"] = base_job_name
                    result["Nodes"] = len(list(job.nodes()))
                    result["Deadline"] = args.deadline
                    result["Percentile"] = args.deadline_percentile
                    result["Algorithm"] = alg
                    result["SchedulingTime"] = sched_time
                    result["AvgCost"] = eval_results.avg_cost
                    result["AvgMakespan"] = eval_results.avg_makespan
                    result["StdCost"] = eval_results.std_cost
                    result["StdMakespan"] = eval_results.std_makespan
                    result["HitRatio"] = eval_results.hit_ratio
                    result["AvgTardiness"] = eval_results.avg_tardiness
                    result["MCRuns"] = eval_results.total_runs
                    result["MCUnfeasibleRuns"] = eval_results.unfeasible_runs
                    for ip,p in enumerate(eval_results.percentiles):
                        result[f"Makespan-P{p}"] = eval_results.makespan_quantiles[ip]
                    for ip,p in enumerate(eval_results.percentiles):
                        result[f"Cost-P{p}"] = eval_results.cost_quantiles[ip]

                    results.append(result)
                    print(result)
                    # save partial results
                    resultsDf = pd.DataFrame(results)
                    if old_results is not None:
                        resultsDf = pd.concat([old_results, resultsDf])
                    resultsDf.to_csv(outfile, index=False)
    
    resultsDf = pd.DataFrame(results)
    if old_results is not None:
        resultsDf = pd.concat([old_results, resultsDf])
    resultsDf.to_csv(outfile, index=False)

def experiment_dummy_workflows (args, debug=False):
    results = []
    args.mc_stopping_rel_error = 0.05
    args.billing_period_sec = 1
    args.moheft_k = 10
    args.genetic_max_evaluations = 50000
    args.percentiles_to_try = 8
    outfile="resultsDummyWorkflows.csv"

    SEEDS=[1,293,287844,2902,944,9573,102903,193,456,71]
    JOBS = ["dummy-ndvi", "dummy-preprocessing", "dummy-firedetection"]
    DEADLINES = [300, 600, 900]
    assert(len(DEADLINES)==len(JOBS))

    # Check existing results
    try:
        old_results = pd.read_csv(outfile)
        # TODO: temporary hack
        if not "Seed" in old_results.columns:
            old_results.loc[:,"Seed"] = SEEDS[0]
        if not "ConservativeMargin" in old_results.columns:
            old_results.loc[:,"ConservativeMargin"] = 0.0

    except:
        old_results = None

    for seed in SEEDS:
        args.scheduler_seed = seed

        for alg in [algorithms.SCHED_CloudMOHEFT, algorithms.SCHED_ProbMOHEFT, algorithms.SCHED_GC, algorithms.SCHED_HEFT, algorithms.SCHED_GENETIC, algorithms.SCHED_ParallelProbMOHEFT]:
            args.algorithm = alg
            for n_vmtypes in [8]:
                infra = create_provider_infrastructure(n_vmtypes)
                for job_name, deadline in zip(JOBS, DEADLINES):
                    job, mean_exec_times, op_output_mb = create_job(job_name)
                    args.deadline = deadline
                    op_distributions = create_gamma_distributions(mean_exec_times, 1.0)
                    predictor = SimplePredictor(op_distributions, scalability_fun, FAMILY_SPEEDUP, op_output_mb=op_output_mb)

                    for percentile in [0.9]:
                        args.deadline_percentile = percentile

                        # Check if we can skip this run
                        if old_results is not None:
                            if not old_results[(old_results.Algorithm == alg) &\
                                    (old_results.Job == job_name) &\
                                    (old_results.MaxIterations == iterations) &\
                                    (old_results.Seed == seed) &\
                                    (old_results.Deadline == deadline) &\
                                    (old_results.VMTypes == n_vmtypes) &\
                                    (old_results.Percentile == percentile)].empty:
                                print("Skipping conf")
                                continue
                        if alg != "Genetic" and seed != SEEDS[0]:
                            continue # we only change seed for Genetic

                        sol, eval_results, sched_time = experiment.run(infra, job, predictor, args, detailed_results=True)

                        result = {}
                        result["VMTypes"] = n_vmtypes
                        result["Job"] = job_name
                        result["Nodes"] = len(list(job.nodes()))
                        result["Deadline"] = deadline
                        result["Percentile"] = percentile
                        result["Algorithm"] = alg
                        result["Seed"] = seed
                        result["AvgCost"] = eval_results.avg_cost
                        result["AvgMakespan"] = eval_results.avg_makespan
                        result["StdCost"] = eval_results.std_cost
                        result["StdMakespan"] = eval_results.std_makespan
                        result["HitRatio"] = eval_results.hit_ratio
                        result["AvgTardiness"] = eval_results.avg_tardiness
                        result["MCRuns"] = eval_results.total_runs
                        result["MCUnfeasibleRuns"] = eval_results.unfeasible_runs
                        for ip,p in enumerate(eval_results.percentiles):
                            result[f"Makespan-P{p}"] = eval_results.makespan_quantiles[ip]
                        for ip,p in enumerate(eval_results.percentiles):
                            result[f"Cost-P{p}"] = eval_results.cost_quantiles[ip]
                        result["SchedulingTime"] = sched_time

                        results.append(result)
                        print(result)
                    # save partial results
                    resultsDf = pd.DataFrame(results)
                    if old_results is not None:
                        resultsDf = pd.concat([old_results, resultsDf])
                    resultsDf.to_csv(outfile, index=False)

    if debug:
        return
    
    resultsDf = pd.DataFrame(results)
    if old_results is not None:
        resultsDf = pd.concat([old_results, resultsDf])
    resultsDf.to_csv(outfile, index=False)

def experiment_main_comparison(args, debug=False):
    results = []
    args.mc_stopping_rel_error = 0.05
    args.billing_period_sec = 1
    args.moheft_k = 10
    args.percentiles_to_try = 8
    outfile="resultsMainComparison.csv"

    SEEDS=[1,293,287844,2902,944,9573,102903,193,456,71]
    JOBS = ["epigenomics", "cybershake", "montage", "sipht", "ligo"]
    DEADLINES = [900, 900, 900, 2400, 1800]
    CONSERVATIVE_MARGINS=[0.0]
    assert(len(DEADLINES)==len(JOBS))

    # Check existing results
    try:
        old_results = pd.read_csv(outfile)
        # TODO: temporary hack
        if not "Seed" in old_results.columns:
            old_results.loc[:,"Seed"] = SEEDS[0]
        if not "ConservativeMargin" in old_results.columns:
            old_results.loc[:,"ConservativeMargin"] = 0.0

    except:
        old_results = None

    MAX_ITERATIONS=[50000, 100000]
    
    for iter_i, iterations in enumerate(MAX_ITERATIONS):
        args.genetic_max_evaluations = iterations
        args.dyna_max_iterations = iterations

        for seed in SEEDS:
            args.scheduler_seed = seed

            for percentile_margin in CONSERVATIVE_MARGINS:
                args.percentile_margin = percentile_margin

                for alg in [algorithms.SCHED_CloudMOHEFT, algorithms.SCHED_ProbMOHEFT, algorithms.SCHED_GC, algorithms.SCHED_HEFT, algorithms.SCHED_GENETIC, algorithms.SCHED_DYNA, algorithms.SCHED_ParallelProbMOHEFT]:
                    args.algorithm = alg
                    for n_vmtypes in [2,4,8,13,21]:
                        infra = create_provider_infrastructure(n_vmtypes)
                        for job_name, deadline in zip(JOBS, DEADLINES):
                            job, mean_exec_times, op_output_mb = create_job(job_name)
                            args.deadline = deadline
                            op_distributions = create_gamma_distributions(mean_exec_times, 1.0)
                            predictor = SimplePredictor(op_distributions, scalability_fun, FAMILY_SPEEDUP, op_output_mb=op_output_mb)

                            if debug and (alg != algorithms.SCHED_HEFT or n_vmtypes > 8):
                                continue
                            for percentile in [0.75, 0.9, 0.95]:
                                args.deadline_percentile = percentile

                                # Check if we can skip this run
                                if old_results is not None:
                                    if not old_results[(old_results.Algorithm == alg) &\
                                            (old_results.Job == job_name) &\
                                            (old_results.MaxIterations == iterations) &\
                                            (old_results.Seed == seed) &\
                                            (old_results.Deadline == deadline) &\
                                            (old_results.ConservativeMargin == percentile_margin) &\
                                            (old_results.VMTypes == n_vmtypes) &\
                                            (old_results.Percentile == percentile)].empty:
                                        print("Skipping conf")
                                        continue
                                if alg == "Dyna" and n_vmtypes > 8:
                                    continue # XXX
                                if alg != "Dyna" and alg != "Genetic" and iter_i > 0:
                                    continue # we only change iterations for Dyna and Genetic
                                if alg != "Genetic" and seed != SEEDS[0]:
                                    continue # we only change seed for Genetic
                                if alg != algorithms.SCHED_ProbMOHEFT and \
                                        alg != algorithms.SCHED_ParallelProbMOHEFT\
                                        and percentile_margin > 0.0:
                                    continue # we only change percentile margin for ProbMOHEFT


                                sol, eval_results, sched_time = experiment.run(infra, job, predictor, args, detailed_results=True)

                                result = {}
                                result["VMTypes"] = n_vmtypes
                                result["Job"] = job_name
                                result["Nodes"] = len(list(job.nodes()))
                                result["Deadline"] = deadline
                                result["Percentile"] = percentile
                                result["Algorithm"] = alg
                                result["MaxIterations"] = iterations
                                result["ConservativeMargin"] = percentile_margin
                                result["Seed"] = seed
                                result["AvgCost"] = eval_results.avg_cost
                                result["AvgMakespan"] = eval_results.avg_makespan
                                result["StdCost"] = eval_results.std_cost
                                result["StdMakespan"] = eval_results.std_makespan
                                result["HitRatio"] = eval_results.hit_ratio
                                result["AvgTardiness"] = eval_results.avg_tardiness
                                result["MCRuns"] = eval_results.total_runs
                                result["MCUnfeasibleRuns"] = eval_results.unfeasible_runs
                                for ip,p in enumerate(eval_results.percentiles):
                                    result[f"Makespan-P{p}"] = eval_results.makespan_quantiles[ip]
                                for ip,p in enumerate(eval_results.percentiles):
                                    result[f"Cost-P{p}"] = eval_results.cost_quantiles[ip]
                                result["SchedulingTime"] = sched_time

                                results.append(result)
                                print(result)
                            # save partial results
                            resultsDf = pd.DataFrame(results)
                            if old_results is not None:
                                resultsDf = pd.concat([old_results, resultsDf])
                            resultsDf.to_csv(outfile, index=False)

    if debug:
        return
    
    resultsDf = pd.DataFrame(results)
    if old_results is not None:
        resultsDf = pd.concat([old_results, resultsDf])
    resultsDf.to_csv(outfile, index=False)


def experiment_dyna(args, debug=False):
    results = []
    args.mc_stopping_rel_error = 0.05
    args.billing_period_sec = 1
    args.moheft_k = 10
    args.percentiles_to_try = 8
    outfile="resultsDynaNew.csv"

    SEEDS=[1,293,287844,2902,944,9573,102903,193,456,71]
    JOBS = ["epigenomics", "cybershake", "montage", "sipht", "ligo"]
    DEADLINES = [900, 900, 900, 2400, 1800]
    DEADLINE_COEFFS = [1, 5, 10]
    assert(len(DEADLINES)==len(JOBS))

    # Check existing results
    try:
        old_results = pd.read_csv(outfile)
        # TODO: temporary hack
        if not "Seed" in old_results.columns:
            old_results.loc[:,"Seed"] = SEEDS[0]
    except:
        old_results = None

    MAX_ITERATIONS=[50000, 100000]
    
    for iter_i, iterations in enumerate(MAX_ITERATIONS):
        args.genetic_max_evaluations = iterations
        args.dyna_max_iterations = iterations

        for seed in SEEDS:
            args.scheduler_seed = seed

            for deadline_coeff in DEADLINE_COEFFS:
                for alg in [algorithms.SCHED_DYNA, algorithms.SCHED_CloudMOHEFT, algorithms.SCHED_ProbMOHEFT, algorithms.SCHED_GC, algorithms.SCHED_HEFT, algorithms.SCHED_GENETIC, algorithms.SCHED_ParallelProbMOHEFT]:
                    args.algorithm = alg
                    for n_vmtypes in [2,4]:
                        infra = create_provider_infrastructure(n_vmtypes, skip_first_machines=4)
                        for job_name, deadline in zip(JOBS, DEADLINES):
                            deadline *= deadline_coeff
                            job, mean_exec_times, op_output_mb = create_job(job_name)
                            args.deadline = deadline
                            op_distributions = create_gamma_distributions(mean_exec_times, 1.0)
                            predictor = SimplePredictor(op_distributions, scalability_fun, FAMILY_SPEEDUP, op_output_mb=op_output_mb)

                            if debug and (alg != algorithms.SCHED_HEFT or n_vmtypes > 8):
                                continue
                            for percentile in [0.75, 0.9, 0.95]:
                                args.deadline_percentile = percentile

                                # Check if we can skip this run
                                if old_results is not None:
                                    if not old_results[(old_results.Algorithm == alg) &\
                                            (old_results.Job == job_name) &\
                                            (old_results.MaxIterations == iterations) &\
                                            (old_results.Seed == seed) &\
                                            (old_results.Deadline == deadline) &\
                                            (old_results.VMTypes == n_vmtypes) &\
                                            (old_results.Percentile == percentile)].empty:
                                        print("Skipping conf")
                                        continue
                                if alg != "Dyna" and alg != "Genetic" and iter_i > 0:
                                    continue # we only change iterations for Dyna and Genetic
                                if alg != "Genetic" and seed != SEEDS[0]:
                                    continue # we only change seed for Genetic


                                sol, eval_results, sched_time = experiment.run(infra, job, predictor, args, detailed_results=True)

                                result = {}
                                result["VMTypes"] = n_vmtypes
                                result["Job"] = job_name
                                result["Nodes"] = len(list(job.nodes()))
                                result["Deadline"] = deadline
                                result["Percentile"] = percentile
                                result["Algorithm"] = alg
                                result["MaxIterations"] = iterations
                                result["Seed"] = seed
                                result["AvgCost"] = eval_results.avg_cost
                                result["AvgMakespan"] = eval_results.avg_makespan
                                result["StdCost"] = eval_results.std_cost
                                result["StdMakespan"] = eval_results.std_makespan
                                result["HitRatio"] = eval_results.hit_ratio
                                result["AvgTardiness"] = eval_results.avg_tardiness
                                result["MCRuns"] = eval_results.total_runs
                                result["MCUnfeasibleRuns"] = eval_results.unfeasible_runs
                                for ip,p in enumerate(eval_results.percentiles):
                                    result[f"Makespan-P{p}"] = eval_results.makespan_quantiles[ip]
                                for ip,p in enumerate(eval_results.percentiles):
                                    result[f"Cost-P{p}"] = eval_results.cost_quantiles[ip]
                                result["SchedulingTime"] = sched_time

                                results.append(result)
                                print(result)
                            # save partial results
                            resultsDf = pd.DataFrame(results)
                            if old_results is not None:
                                resultsDf = pd.concat([old_results, resultsDf])
                            resultsDf.to_csv(outfile, index=False)

    if debug:
        return
    
    resultsDf = pd.DataFrame(results)
    if old_results is not None:
        resultsDf = pd.concat([old_results, resultsDf])
    resultsDf.to_csv(outfile, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', action='store', required=False, default=None, type=str)
    parser.add_argument('--job', action='store', required=False, default="epigenomics", type=str)
    parser.add_argument('--max_vmtypes', action='store', required=False, default=-1, type=int)
    parser.add_argument('--max_vcpu', action='store', required=False, default=576, type=int)
    parser.add_argument('--moheft_k', action='store', required=False, default=15, type=int)
    parser.add_argument('--deadline', action='store', required=False, default=60.0, type=float)
    parser.add_argument('--deadline_percentile', action='store', required=False, default=0.9, type=float)
    parser.add_argument('--algorithm', action='store', required=False, default="CloudMOHEFT", type=str)
    parser.add_argument('--evaluations', action='store', required=False, default=10000, type=int)
    parser.add_argument('--evaluation_seed', action='store', required=False, default=4123, type=int)
    parser.add_argument('--durations_seed', action='store', required=False, default=1, type=int)
    parser.add_argument('--scheduler_seed', action='store', required=False, default=9487, type=int)
    parser.add_argument('--ntasks', action='store', required=False, default=1, type=int)
    parser.add_argument('--job_as_xmlgraph', action='store', required=False, default=None)
    parser.add_argument('--genetic_max_evaluations', action='store', required=False, default=50000, type=int)
    parser.add_argument('--genetic_population', action='store', required=False, default=1000, type=int)
    parser.add_argument('--accurate_mc', action='store_true', required=False, default=False)
    parser.add_argument('--mc_stopping_rel_error', action='store', required=False, default=0.05, type=float)
    parser.add_argument('--dyna_max_iterations', action='store', required=False, default=2000, type=int)
    parser.add_argument('--billing_period_sec', action='store', required=False, default=0, type=int)
    parser.add_argument('--prob_return_frontier', action='store_true', required=False, default=False)
    parser.add_argument('--percentile_epsilon', action='store', required=False, default=0.02, type=float)
    parser.add_argument('--percentile_margin', action='store', required=False, default=0.0, type=float)
    parser.add_argument('--percentiles_to_try', action='store', required=False, default=-1, type=int)
    parser.add_argument('--profile', action='store_true', required=False, default=False)

    args = parser.parse_args()

    if args.experiment is not None:
        assert(not args.profile)
        if args.experiment.lower() == "c":
            experiment_capacity(args)
        elif args.experiment.lower() == "a":
            experiment_main_comparison(args)
        elif args.experiment.lower() == "t":
            experiment_probmoheft(args)
        elif args.experiment.lower() == "g":
            experiment_genetic(args)
        elif args.experiment.lower() == "g2":
            experiment_genetic2(args)
        elif args.experiment.lower() == "d":
            experiment_distributions(args)
        elif args.experiment.lower() == "s":
            experiment_scalability(args)
        elif args.experiment.lower() == "s2":
            experiment_scalability2(args)
        elif args.experiment.lower() == "z":
            experiment_dyna(args)
        elif args.experiment.lower() == "w":
            experiment_dummy_workflows(args)
        else:
            print("Unknown experiment!")
        exit(0)

    if args.profile:
        import cProfile
        cProfile.run("main(args)", "profilestats")
    else:
        main(args)
