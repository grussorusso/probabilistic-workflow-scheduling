from jmetal.algorithm.multiobjective.nsgaii import NSGAII,DistributedNSGAII
from jmetal.operator.mutation  import CompositeMutation, IntegerPolynomialMutation, NullMutation, Mutation
from jmetal.operator.crossover import CompositeCrossover, IntegerSBXCrossover, NullCrossover
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.evaluator import MultiprocessEvaluator, SequentialEvaluator
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solution import get_non_dominated_solutions#, print_function_values_to_file, print_variables_to_file
from jmetal.util.generator import InjectorGenerator

from jmetal.core.problem import Problem
from jmetal.core.solution import  CompositeSolution, IntegerSolution
from jmetal.core.operator import Crossover

from jmetal.util.ckecking import Check

import random
import os
import copy
import networkx as nx

from scheduler.job import *
from scheduler.infrastructure import *
from scheduler.scheduling import SchedulingSolution, ScheduleEntry
from scheduler import evaluation as monte_carlo

# ----------------------------------------- sim.py

def get_time(machine, time, machines):
	return time/machines[machine].core_count

def get_cost(machine, time, machines):
	return time*machines[machine].cost/machines[machine].core_count


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
class TopologySPXCrossover(Crossover[IntegerSolution, IntegerSolution]):
    def __init__(self, probability: float):
        super(TopologySPXCrossover, self).__init__(probability=probability)

    def execute(self, parents: list[IntegerSolution]) -> list[IntegerSolution]:
        Check.that(type(parents[0]) is IntegerSolution, "Solution type invalid")
        Check.that(type(parents[1]) is IntegerSolution, "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))

        offspring = copy.deepcopy(parents)
        rand = random.random()

        if rand <= self.probability:
            # 1. Get the total number of variables
            total_number_of_vars = len(parents[0].variables)

            # 2. Calculate the point to make the crossover
            crossover_point = random.randrange(total_number_of_vars)

            # 3. first loop
            count0 = crossover_point+1
            count1 = crossover_point+1
            for i in range(total_number_of_vars):
                if parents[1].variables[i] not in offspring[0].variables:  
                    offspring[0].variables[count0] = parents[1].variables[i]
                    count0+=1
                if parents[0].variables[i] not in offspring[1].variables:  
                    offspring[1].variables[count1] = parents[0].variables[i]
                    count1+=1

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Single point topology crossover"



class TopologyMutation(Mutation[IntegerSolution]):
    def __init__(self, probability: float, distribution_index: float = 0.20, predecessors=None, successors=None):
        super(TopologyMutation, self).__init__(probability=probability)
        Check.that(predecessors != None, "predecessors deps invalid")
        Check.that(successors != None, "successors deps invalid")

        self.distribution_index = distribution_index
        self.predecessors = predecessors
        self.successors = successors

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        Check.that(issubclass(type(solution), IntegerSolution), "Solution type invalid")

        original = copy.deepcopy(solution)

        for i in range(solution.number_of_variables):
            if random.random() <= self.probability:
                start = i
                end = i
                item = solution.variables[i]

                while start >= 0 and solution.variables[start] not in self.predecessors[item]:
                    start -= 1 

                while end < solution.number_of_variables and solution.variables[end] not in self.successors[item]:
                    end += 1 

                start+=1
                pos = random.randrange(start, end)

                if pos == i:
                    continue
                offset = 0
                if pos > i:
                    offset-=1

                solution.variables.pop(i) 
                solution.variables.insert(pos+offset, item) 

        return solution

    def get_name(self):
        return "Topology mutation (Integer)"




class SchedulingCVN(Problem):

   def __init__(self, graph, machines, deadline, deadline_percentile, node_list, mc_evaluator):
      super(SchedulingCVN, self)
      self.number_of_variables = 2

      self.temp=0

      self.number_of_objectives = 4
      self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
      self.obj_labels = ['Deadline', 'Cost', 'NConstrV', 'ConstrV']

      self.number_of_constraints = 0

      self.graph = graph
      self.machines = machines
      self.deadline = deadline
      self.deadline_percentile = deadline_percentile
      self.node_list = node_list
      self.mc_evaluator = mc_evaluator
      self.number_of_tasks = graph.number_of_nodes()
      self.max_cost,self.min_power = get_mostexpensive_slowest_machine(machines)
      self.count = 0


   def initial_solutions(self,size, all_topo_sort):
      res = []
      a_lower_bound =  [0]*self.number_of_tasks
      b_lower_bound =  [0]*self.number_of_tasks
      a_upper_bound =  [len(self.machines)*self.number_of_tasks]*(self.number_of_tasks)
      b_upper_bound =  [self.number_of_tasks]*(self.number_of_tasks)

      #topological_sort = random.choice(all_topo_sort)
      topological_sort = next(all_topo_sort)

      for i in range(4):
        a = IntegerSolution(lower_bound=a_lower_bound,upper_bound=a_upper_bound,number_of_objectives=self.number_of_objectives)
        b = IntegerSolution(lower_bound=b_lower_bound,upper_bound=b_upper_bound,number_of_objectives=self.number_of_objectives)
        res += [CompositeSolution([a, b])]

      #TODO get topological sort
      res[0].variables[0].variables = [self.max_cost  +i*len(self.machines) for i in range(self.number_of_tasks)] #N most expensive instances
      res[1].variables[0].variables = [self.min_power +i*len(self.machines) for i in range(self.number_of_tasks)] #N slowest instances
      res[2].variables[0].variables = [self.max_cost  +i*len(self.machines) for i in range(self.number_of_tasks)] #1 most expensive instance
      res[3].variables[0].variables = [self.min_power +i*len(self.machines) for i in range(self.number_of_tasks)] #1 slowest instance
      for i in range(4):
        res[i].variables[1].variables = [x for x in topological_sort]

      # XXX
      self.some_topo_sorts = []
      self.sort_index = 0
      i=0
      while i < size - len(res):
          try:
              self.some_topo_sorts.append(next(all_topo_sort))
              i+=1
          except StopIteration:
              all_topo_sort = nx.all_topological_sorts(self.graph)

      return res 

   def sol2sched_solution (self, solution: CompositeSolution) -> SchedulingSolution:
      assigments = {}
      # Maps each VM type to the set of used instances (i.e., integers)
      vmtype2instances = {}


      for i in range(self.number_of_tasks):
          instance_id = solution.variables[0].variables[i]
          vmtype = self.machines[instance_id %  len(self.machines)]
          if not vmtype in vmtype2instances: 
              vmtype2instances[vmtype] = set()
          vmtype2instances[vmtype].add(instance_id)

          task     = self.node_list[solution.variables[1].variables[i]]
          if instance_id not in assigments: assigments[instance_id] = []
          #print(instance, task)
          assigments[instance_id].append(ScheduleEntry(task, len(assigments[instance_id]))) 

      # Create the actual VM instances
      vm_schedule = {}
      task2vm = {}
      for vmtype in vmtype2instances:
          for i,id in enumerate(vmtype2instances[vmtype]):
              vm = (vmtype, i)
              vm_schedule[vm] = assigments[id]

              for entry in assigments[id]:
                  task2vm[entry.task] = vm

      sched_solution = SchedulingSolution()
      sched_solution.subtask2instance = task2vm
      sched_solution.vm_schedule = vm_schedule
      return sched_solution
   
   def evaluate(self, solution:  CompositeSolution) ->  CompositeSolution:
    self.count+=1
    sched_solution = self.sol2sched_solution(solution)

    hits, mean_makespan, mean_cost, makespan_quantile = self.mc_evaluator.run(sched_solution)

    constraint_violation = 0
    if hits < self.deadline_percentile:
        constraint_violation = 1
    if constraint_violation > 0:
        violation_amount = (makespan_quantile - self.deadline)/self.deadline
    else:
        violation_amount = 0


    solution.objectives[0] = mean_makespan
    solution.objectives[1] = mean_cost
    solution.objectives[2] = constraint_violation
    solution.objectives[3] = violation_amount

    return solution



   def create_solution(self) -> CompositeSolution:

        #topological_sort = random.choice(self.all_topo_sort)
        topological_sort = self.some_topo_sorts[self.sort_index]
        self.sort_index = (self.sort_index + 1) % len(self.some_topo_sorts)

        a_lower_bound =  [0]*self.number_of_tasks
        b_lower_bound =  [0]*self.number_of_tasks
        a_upper_bound =  [len(self.machines)*self.number_of_tasks]*(self.number_of_tasks)
        b_upper_bound =  [self.number_of_tasks]*(self.number_of_tasks)
        a = IntegerSolution(lower_bound=a_lower_bound,upper_bound=a_upper_bound,number_of_objectives=self.number_of_objectives)
        b = IntegerSolution(lower_bound=b_lower_bound,upper_bound=b_upper_bound,number_of_objectives=self.number_of_objectives)
        res = CompositeSolution([a, b])
        res.variables[0].variables = [random.randrange(len(self.machines)*self.number_of_tasks)  for i in range(self.number_of_tasks)]
        res.variables[1].variables = [x for x in topological_sort]
        return res

   def get_name(self) -> str:
      return 'CalzarossaVedovaNebbione Scheduling'

class GeneticScheduler:

    def __init__ (self, infra, pred, max_evaluations, deadline_percentile=0.9, 
                  population_size=1000, algorithm_seed=1234, accurate_monte_carlo=False,
                  mc_stopping_rel_error=0.1, return_frontier=False):
        self.infrastructure = infra
        self.predictor = pred
        self.deadline_percentile = deadline_percentile
        self.max_evaluations = max_evaluations
        self.population = population_size
        self.algorithm_seed = algorithm_seed
        self.accurate_monte_carlo = accurate_monte_carlo
        self.mc_stopping_rel_error = mc_stopping_rel_error
        self.return_frontier = return_frontier

        self.parallel = False

        if self.parallel:
           import dask
           from dask.distributed import Client
           from distributed import LocalCluster
           dask.config.set(scheduler='processes')
           self.client = Client(n_workers=20, threads_per_worker=1)
           self.ncores = sum(client.ncores().values())

    def schedule (self, job, deadline):
        assert(isinstance(job, Job))
        assert(deadline > 0.0)

        random.seed(self.algorithm_seed)

        population_size = self.population
        offspring_population_size = 1000 # TODO

        node_mapping = {}
        node_list = []
        count=0
        for node in job.nodes:
            node_mapping[node] = count
            node_list.append(node)
            count+=1
        graph = nx.relabel_nodes(job, node_mapping, copy=True)

        vm_types = list(self.infrastructure.vm_types)

        mc_evaluator = monte_carlo.MonteCarloEvaluator(job, self.predictor,
                        deadline, self.deadline_percentile,
                        max_relative_error=self.mc_stopping_rel_error, change_seed=False,
                        verbose=False, batch_size=100, accurate_simulation=self.accurate_monte_carlo)


        problem=SchedulingCVN(graph, vm_types, deadline, self.deadline_percentile, node_list, mc_evaluator)
        all_topo_sorts = nx.all_topological_sorts(graph)
        initial_solutions = problem.initial_solutions(population_size,all_topo_sorts)
        generator=InjectorGenerator(solutions=initial_solutions)

        successors = {}
        predecessors = {}
        for n in graph.nodes():
            predecessors[n] = set(list(graph.predecessors(n)))
            successors[n]   = set(list(graph.successors(n)))


        if self.parallel:
            algorithm =  DistributedNSGAII(
                                problem=problem,
                                population_size=population_size,
                                offspring_population_size=offspring_population_size,
                                mutation=CompositeMutation([IntegerPolynomialMutation(0.02, 20), NullMutation()]),
                                crossover=CompositeCrossover([IntegerSBXCrossover(probability=0.9, distribution_index=20),NullCrossover()]),
                                termination_criterion=StoppingByEvaluations(max_evaluations=self.max_evaluations),
                                population_generator = generator,
                                number_of_cores=ncores,
                                client=client
                                )
        else:
            algorithm =  NSGAII(
                                problem=problem,
                                population_size=population_size,
                                offspring_population_size=offspring_population_size,
                                mutation=CompositeMutation([IntegerPolynomialMutation(0.02, 20), TopologyMutation(0.02, predecessors=predecessors, successors=successors)]),
                                crossover=CompositeCrossover([IntegerSBXCrossover(probability=0.9, distribution_index=20),TopologySPXCrossover(probability=0.9)]),
                                termination_criterion=StoppingByEvaluations(max_evaluations=self.max_evaluations),
                                population_evaluator = MultiprocessEvaluator(processes=os.cpu_count()),
                                #population_evaluator = SequentialEvaluator(),
                                population_generator = generator,
                                )
        progress_bar = ProgressBarObserver(max=self.max_evaluations)
        algorithm.observable.register(progress_bar)
        algorithm.run()


        front = get_non_dominated_solutions(algorithm.get_result())
        if self.return_frontier:
            return [problem.sol2sched_solution(sol) for sol in front]
        
        # pick cheapest among solutions not violating constraint (if any)
        cheapest_sol = None
        min_cost = front[0].objectives[1]
        for solution in front:
            if solution.objectives[2] == 0 and solution.objectives[1] <= min_cost:
                cheapest_sol = solution
                min_cost = cheapest_sol.objectives[1]
        if cheapest_sol is None:
            print("Picking solution that minimizes violations...")
            # pick sol minimizing violations
            min_viol = front[0].objectives[3]
            for solution in front:
                if solution.objectives[3] <= min_viol:
                    cheapest_sol = solution
                    min_viol = solution.objectives[3]


        return problem.sol2sched_solution(cheapest_sol)

