from scheduler.job import *
from scheduler.infrastructure import *
from scheduler.scheduling import SchedulingSolution, ScheduleEntry
from scheduler.heft import HEFT

class GreedyCost(HEFT):

    def __init__ (self, infra, pred):
        super().__init__(infra, pred)

    def schedule (self, job, deadline):
        assert(isinstance(job, Job))
        assert(deadline > 0.0)


        I = list(self.infra.all_instances())

        if self.sol is None:
            self.sol = SchedulingSolution()
            for i in I:
                self.sol.vm_schedule[i] = []

        aft = {}
        rankU = self.compute_rankU(job)
        tasklist = sorted(job.nodes, key=lambda n: rankU[n], reverse=True)

        while len(tasklist) > 0:
            n = tasklist[0]
            tasklist = tasklist[1:]

            # Scheduling n
            min_cost = None
            min_eft = None
            min_est = None
            min_eft_vm = None
            for vm in I:
                # Compute EST based on predecessors (Eq. 5, HEFT)
                est = 0.0
                data_reading_time = 0.0
                for p in job.predecessors(n):
                    est = max(est, aft[p])
                    if self.sol.subtask2instance[p] != vm:
                        data_reading_time = max(data_reading_time, self.pred.data_reading_time(p[0],n[0],vm[0]))

                # Compute EST based on availability (Eq. 5, HEFT)
                # Insertion-based policy
                est, first_on_the_machine = self.min_schedulable_time (n, vm, self.sol.vm_schedule[vm], job, est)
                first_in_the_graph = (len(list(job.predecessors(n)))==0)

                # EFT (Eq. 6, HEFT)

                exec_time = self.pred.exec_time(n[0], job, vm[0], first_on_the_machine=first_on_the_machine, first_in_the_graph=first_in_the_graph) +\
                        data_reading_time +\
                        self.pred.data_writing_time(n[0], vm[0]) # We are assuming that successors are not co-located here...
                eft = est + exec_time
                cost = vm[0].cost * exec_time

                if min_cost is None or cost < min_cost:
                    min_est = est
                    min_eft = eft
                    min_eft_vm = vm
                    min_cost = cost

            aft[n] = min_eft
            print(f"Scheduling {n} to {min_eft_vm} starting at {min_est}")
            self.sol.add_scheduled_subtask(n, min_eft_vm, min_est, min_eft)

        self.fix_schedule_with_colocation(job, rankU, aft, self.sol)
        return self.sol.copy()

