from scheduler.job import Operator,Job
from scheduler.vmprovider import VMType

class UniversalScalabilityFunction:
    def __init__ (self, alpha, beta=0.0, single_core_speedup=1.0, name=None):
        self.alpha = alpha
        self.beta = beta
        self.single_core_speedup = single_core_speedup

        if name is None:
            self.name = f"USF-{alpha}-{beta}"
        else:
            self.name = name

    def eval(self, n):
        return (n / (1 + self.alpha*(n-1) + self.beta*n*(n-1))) * self.single_core_speedup

    def print(self):
        for cores in range(1,20):
            print(f"{cores}: {self.eval(cores):.2f}")

# Just a toy implementation
class Predictor:

    def exec_time (self, op: Operator, job: Job, vm_type: VMType, first_on_the_machine=False, first_in_the_graph=False):
        if vm_type.core_count == 4:
            t = 2.0
        else:
            t = 2.5

        t += self.setup_overhead(vm_type, first_on_the_machine, first_in_the_graph)
        return t

    def setup_overhead (self, vm_type, first_on_the_machine, first_in_the_graph):
        if first_on_the_machine:
            return 0.5
        elif first_in_the_graph:
            return 0.25
        else:
            return 0.0

    def data_writing_time (self, op, vm_type=None):
        return 0.1 # TODO

    def data_reading_time (self, op1, op2, vm_type=None):
        return 0.1 # TODO

    def get_exec_time_distribution (self, op: Operator, job: Job, vm_type: VMType, first_on_the_machine=False, first_in_the_graph=False):
        return None

class SimplePredictor:

    # op_distributions: dict
    # family_speedup: speedup factor for each VM family
    def __init__ (self, op_distributions: dict, scalability_fun: UniversalScalabilityFunction, family_speedup=None,
                  op_output_mb={}, vm_startup_time=30):
        self.op_distributions = op_distributions
        self.scalability_fun = scalability_fun
        self.family_speedup = family_speedup
        self.op_output_mb = op_output_mb
        self.vm_startup_time = vm_startup_time

    def exec_time (self, op: Operator, job: Job, vm_type: VMType, first_on_the_machine=False, first_in_the_graph=False):
        speedup = self.scalability_fun.eval(vm_type.core_count)
        if self.family_speedup is not None and vm_type.family in self.family_speedup:
            speedup *= self.family_speedup[vm_type.family]
        t = self.op_distributions[op].get_mean()/speedup

        t += self.setup_overhead(vm_type, first_on_the_machine, first_in_the_graph)
        return t

    def setup_overhead (self, vm_type, first_on_the_machine, first_in_the_graph):
        if first_on_the_machine:
            return self.vm_startup_time
        return  0.0

    def data_writing_time (self, op, vm_type=None):
        default_bw = 12.5 # MB/s = 100 Mbit/s
        default_size = 1.0

        if vm_type is not None:
            bw = vm_type.bandwidth_MBs
            if bw is None or bw <= 0.0:
                bw = default_bw
        size = self.op_output_mb.get(op, default_size)
        return size/bw

    def data_reading_time (self, op1, op2, vm_type=None):
        default_bw = 12.5 # MB/s = 100 Mbit/s
        default_size = 1.0
        if vm_type is not None:
            bw = vm_type.bandwidth_MBs
            if bw is None or bw <= 0.0:
                bw = default_bw
        size = self.op_output_mb.get(op1, default_size)
        return size/bw

    def get_exec_time_distribution (self, op: Operator, job: Job, vm_type: VMType, first_on_the_machine=False, first_in_the_graph=False):
        # rescale distribution based on predicted avg exec time
        avg_exec_time = self.exec_time(op, job, vm_type, first_on_the_machine, first_in_the_graph)
        return self.op_distributions[op].rescaled(avg_exec_time)

