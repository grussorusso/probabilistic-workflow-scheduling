import random

from scheduler.job import *
from scheduler.infrastructure import *
from scheduler.scheduling import SchedulingSolution, ScheduleEntry

class RandomHeuristic:

    def __init__ (self, infra, pred):
        self.infra = infra
        self.pred = pred
        self.sol = None

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
            min_eft = None
            min_est = None
            min_eft_vm = None
            for vm in [random.choice(I)]:
                # Compute EST based on predecessors (Eq. 5, HEFT)
                est = 0.0
                data_reading_time = 0.0
                for p in job.predecessors(n):
                    est = max(est, aft[p])
                    if self.sol.subtask2instance[p] != vm:
                        data_reading_time = max(data_reading_time, self.pred.data_reading_time(p[0],n[0],vm[0]))

                # Compute EST based on availability (Eq. 5, HEFT)
                # Insertion-based policy
                est, first_on_the_machine = self.min_schedulable_time (n, vm, self.sol.vm_schedule[vm], est)
                first_in_the_graph = (len(list(job.predecessors(n)))==0)

                # EFT (Eq. 6, HEFT)

                exec_time = self.pred.exec_time(n[0], job, vm[0], first_on_the_machine=first_on_the_machine, first_in_the_graph=first_in_the_graph) +\
                        data_reading_time +\
                        self.pred.data_writing_time(n[0], vm[0]) # We are assuming that successors are not co-located here...
                eft = est + exec_time

                if min_eft is None or eft < min_eft:
                    min_est = est
                    min_eft = eft
                    min_eft_vm = vm

            aft[n] = min_eft
            print(f"Scheduling {n} to {min_eft_vm} starting at {min_est}")
            self.sol.add_scheduled_subtask(n, min_eft_vm, min_est, min_eft)

        self.fix_schedule_with_colocation(job, rankU, aft, self.sol)
        return self.sol.copy()

    def min_schedulable_time (self, n, vm, curr_schedule, t0=0.0):
        # sort scheduled slots
        schedule = sorted(curr_schedule)

        # machine is idle
        if len(schedule) == 0:
            return t0, True

        exec_time = self.pred.exec_time(n[0], job, vm[0])

        # try to schedule as first
        if t0 + exec_time < schedule[0].est:   
            return t0, True

        # search for a gap between entries i and i+1
        for i in range(len(schedule)-1):
            s = schedule[i]
            s1 = schedule[i+1]
            gap = schedule[i+1].est - max(t0,schedule[i].eft)
            if gap >= exec_time:
                return max(t0,schedule[i].eft), False # Found a gap

        # schedule as last
        return max(t0,schedule[-1].eft), False


    def avg_computation_cost (self, node):
        exec_times = [self.pred.exec_time(node[0], job, vmt) for vmt in self.infra.vm_types]
        return sum(exec_times)/len(exec_times)

    def avg_communication_cost (self, node1, node2):
        costs = []
        for t1 in self.infra.vm_types:
            for t2 in self.infra.vm_types:
                costs.append(self.pred.data_writing_time(node1[0], t1) + \
                   self.pred.data_reading_time(node1[0], node2[0], t2))
        return sum(costs)/len(costs)

    def compute_rankU (self, job):
        ru = {n: 0.0 for n in list(job.nodes)}
        for snk in job.sinks():
            tree = nx.bfs_tree(job, snk, reverse=True)
            for n in tree:
                succ_rank = [self.avg_communication_cost(n,succ) + ru[succ] for succ in job.successors(n)]
                if len(succ_rank) == 0:
                    succ_rank = [0.0]
                ru[n] = self.avg_computation_cost(n) + max(succ_rank)
        return ru

    def fix_schedule_with_colocation (self, job, rankU, aft, sol):
        """
        We exploit co-location if possible to avoid some result writing,
        whose delay has been considered during HEFT execution.
        """
        tasklist = sorted(job.nodes, key=lambda n: rankU[n], reverse=True)

        while len(tasklist) > 0:
            n = tasklist[0]
            tasklist = tasklist[1:]

            vm = sol.subtask2instance[n]
            sched = sol.vm_schedule[vm]
            for entry in sched:
                if entry.task == n:
                    current_est = entry.est
                    break

            # Compute EST based on predecessors (Eq. 5, HEFT)
            est = 0.0
            data_reading_time = 0.0
            for p in job.predecessors(n):
                est = max(est, aft[p])
                if sol.subtask2instance[p] != vm:
                    data_reading_time = max(data_reading_time, self.pred.data_reading_time(p[0],n[0],vm[0]))

            # Try to move this task
            earlier_completions = [e.eft for e in sched if e.eft <= current_est]
            if len(earlier_completions) > 0:
                est = max(est, max(earlier_completions))

            first_on_the_machine = len(earlier_completions) == 0
            first_in_the_graph = len(list(job.predecessors(n))) == 0
            colocated_successors = True
            for p in job.successors(n):
                if sol.subtask2instance[p] != vm:
                    colocated_successors = False

            exec_time = self.pred.exec_time(n[0], job, vm[0], first_on_the_machine=first_on_the_machine,
                    first_in_the_graph=first_in_the_graph) + data_reading_time
            if not colocated_successors:
                    exec_time += self.pred.data_writing_time(n[0], vm[0]) 
            eft = est + exec_time

            # Update schedule and AFT
            aft[n] = eft # Update
            for i in range(len(sched)):
                if sched[i].task == n:
                    sched[i] = ScheduleEntry(n,est,est+exec_time)
                    break
