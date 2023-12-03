import networkx as nx

from scheduler.scheduling import *
import scheduler.heft as heft
import scheduler.greedy as greedy
import scheduler.genetic as genetic
import scheduler.dyna as dyna
from scheduler import baselines
from scheduler.probabilistic import ProbabilisticMOHEFT, ParallelProbMOHEFT, ParallelProbMOHEFT2

SCHED_HEFT="HEFT"
SCHED_GC="GreedyCost"
SCHED_MOHEFT="MOHEFT"
SCHED_CloudMOHEFT="CloudMOHEFT"
SCHED_ProbMOHEFT="ProbMOHEFT"
SCHED_ParallelProbMOHEFT="ParallelProbMOHEFT"
SCHED_ParallelProbMOHEFT2="ParallelProbMOHEFT2"
SCHED_RANDOM="Random"
SCHED_GENETIC="Genetic"
SCHED_DYNA="Dyna"

class Scheduler:

    def __init__ (self, infrastructure):
        self.infrastructure = infrastructure

    def schedule (self, job, predictor, deadline, percentile=-1.0, algorithm="CloudMOHEFT",
                  other_opts=None):
        assert(nx.is_directed_acyclic_graph(job))

        if algorithm == SCHED_CloudMOHEFT:
            h = heft.CloudMOHEFT(self.infrastructure, predictor, K=other_opts.moheft_k)
            sol = h.schedule(job, deadline)
        elif algorithm == SCHED_HEFT or algorithm == "heft":
            h = heft.HEFT(self.infrastructure, predictor)
            sol = h.schedule(job, deadline)
        elif algorithm == SCHED_GC:
            h = greedy.GreedyCost(self.infrastructure, predictor)
            sol = h.schedule(job, deadline)
        elif algorithm == SCHED_MOHEFT:
            h = heft.MOHEFT(self.infrastructure, predictor, K=other_opts.moheft_k)
            sol = h.schedule(job, deadline)
        elif algorithm == SCHED_ProbMOHEFT:
            ph = ProbabilisticMOHEFT(self.infrastructure, predictor, K=other_opts.moheft_k,
                                     required_percentile=percentile,
                                         accurate_monte_carlo = other_opts.accurate_mc,
                                     percentile_stopping_threshold=other_opts.percentile_epsilon,
                                     conservative_hit_ratio_margin=other_opts.percentile_margin,
                                     return_frontier=other_opts.prob_return_frontier,
                                         mc_stopping_rel_error = other_opts.mc_stopping_rel_error)
            sol = ph.schedule(job, deadline)
        elif algorithm == SCHED_ParallelProbMOHEFT:
            ph = ParallelProbMOHEFT(self.infrastructure, predictor, K=other_opts.moheft_k,
                                     required_percentile=percentile,
                                         accurate_monte_carlo = other_opts.accurate_mc,
                                     conservative_hit_ratio_margin=other_opts.percentile_margin,
                                    percentiles_count=other_opts.percentiles_to_try,
                                     return_frontier=other_opts.prob_return_frontier,
                                         mc_stopping_rel_error = other_opts.mc_stopping_rel_error)
            sol = ph.schedule(job, deadline)
        elif algorithm == SCHED_ParallelProbMOHEFT2:
            ph = ParallelProbMOHEFT2(self.infrastructure, predictor, K=other_opts.moheft_k,
                                     required_percentile=percentile,
                                         accurate_monte_carlo = other_opts.accurate_mc,
                                     conservative_hit_ratio_margin=other_opts.percentile_margin,
                                    percentiles_count=other_opts.percentiles_to_try,
                                     return_frontier=other_opts.prob_return_frontier,
                                         mc_stopping_rel_error = other_opts.mc_stopping_rel_error)
            sol = ph.schedule(job, deadline)
        elif algorithm == SCHED_RANDOM:
            ph = baselines.RandomHeuristic(self.infrastructure, predictor)
            sol = ph.schedule(job, deadline)
        elif algorithm == SCHED_DYNA:
            ph = dyna.DynaScheduler(self.infrastructure, predictor, other_opts.dyna_max_iterations, deadline_percentile=percentile,
                                         mc_stopping_rel_error = other_opts.mc_stopping_rel_error)
            sol = ph.schedule(job, deadline)
        elif algorithm == SCHED_GENETIC:
            try:
                max_evals = other_opts.genetic_max_evaluations
            except:
                max_evals = 50000
            s = genetic.GeneticScheduler(self.infrastructure, predictor, max_evals, deadline_percentile=percentile,
                                         algorithm_seed = other_opts.scheduler_seed,
                                         accurate_monte_carlo = other_opts.accurate_mc,
                                         mc_stopping_rel_error = other_opts.mc_stopping_rel_error,
                                         population_size=other_opts.genetic_population)
            sol = s.schedule(job, deadline)
        else:
            raise RuntimeError(f"Unknown algorithm: {algorithm}")

        if sol is None:
            print("No feasible solution!")
            return None

        return sol
