import numpy as np
from scheduler.job import *
from scheduler.infrastructure import *
from scheduler.scheduling import SchedulingSolution, ScheduleEntry

class ICPCP:

    def __init__ (self, infra, pred):
        self.infra = infra
        self.pred = pred
        self.sol = None

    def is_virtual_node(self, node):
        """Check if a node is a virtual entry or exit node."""
        return node[0].name in ["virtual_entry", "virtual_exit"]

    def schedule (self, job, deadline):
        assert(isinstance(job, Job))
        assert(deadline > 0.0)

        # Clone the job and ensure single entry and exit nodes
        job = job.copy()
        
        # Get sources and sinks
        sources = list(job.sources())
        sinks = list(job.sinks())
        
        # Add virtual entry node if there are multiple sources
        virtual_entry = None
        if len(sources) > 1:
            virtual_entry = (Operator("virtual_entry"), -1)
            job.add_node(virtual_entry)
            for src in sources:
                job.add_edge(virtual_entry, src)
        
        # Add virtual exit node if there are multiple sinks
        virtual_exit = None
        if len(sinks) > 1:
            virtual_exit = (Operator("virtual_exit"), -1)
            job.add_node(virtual_exit)
            for sink in sinks:
                job.add_edge(sink, virtual_exit)

        I = list(self.infra.all_instances())

        if self.sol is None:
            self.sol = SchedulingSolution()
            for i in I:
                self.sol.vm_schedule[i] = []

        aft = {}
        est, eft, lft = self.compute_est_eft_lft(job, deadline)
        if np.any(np.array(list(lft.values())) < 0):
            print("Job cannot be scheduled within the given deadline.")
            return None
        tasklist = sorted(job.nodes, key=lambda n: eft[n], reverse=True)

        while len(tasklist) > 0:
            n = tasklist[0]
            tasklist = tasklist[1:]

            # Scheduling n
            min_actual_eft = None
            min_actual_est = None
            min_eft_vm = None
            for vm in I:
                # Compute actual EST based on predecessors' actual finish times
                actual_est = 0.0
                data_reading_time = 0.0
                for p in job.predecessors(n):
                    actual_est = max(actual_est, aft[p])
                    if not self.is_virtual_node(n) and not self.is_virtual_node(p) and self.sol.subtask2instance[p] != vm:
                        data_reading_time = max(data_reading_time, self.pred.data_reading_time(p[0],n[0],vm[0]))

                # Use the pre-computed EST as a lower bound
                actual_est = max(actual_est, est[n])

                # Compute EST based on availability (Eq. 5, HEFT)
                # Insertion-based policy
                actual_est, first_on_the_machine = self.min_schedulable_time (n, vm, self.sol.vm_schedule[vm], job, actual_est)
                first_in_the_graph = (len(list(job.predecessors(n)))==0)

                # EFT (Eq. 6, HEFT)
                # Virtual nodes have zero execution time
                if self.is_virtual_node(n):
                    exec_time = 0.0
                else:
                    exec_time = self.pred.exec_time(n[0], job, vm[0], first_on_the_machine=first_on_the_machine, first_in_the_graph=first_in_the_graph) +\
                            data_reading_time +\
                            self.pred.data_writing_time(n[0], vm[0]) # We are assuming that successors are not co-located here...
                actual_eft = actual_est + exec_time

                if min_actual_eft is None or actual_eft < min_actual_eft:
                    min_actual_est = actual_est
                    min_actual_eft = actual_eft
                    min_eft_vm = vm


            aft[n] = min_actual_eft
            print(f"Scheduling {n} to {min_eft_vm} starting at {min_actual_est}")
            self.sol.add_scheduled_subtask(n, min_eft_vm, min_actual_est, min_actual_eft)


        self.fix_schedule_with_colocation(job, eft, aft, self.sol)
        return self.sol.copy()

    def min_schedulable_time (self, n, vm, curr_schedule, job, t0=0.0):
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


    def avg_computation_cost (self, node, job):
        if self.is_virtual_node(node):
            return 0.0
        exec_times = [self.pred.exec_time(node[0], job, vmt) for vmt in self.infra.vm_types]
        return sum(exec_times)/len(exec_times)

    def min_computation_cost (self, node, job):
        """
        Returns the execution time of a node on the best (fastest) possible VM.
        """
        if self.is_virtual_node(node):
            return 0.0
        exec_times = [self.pred.exec_time(node[0], job, vmt) for vmt in self.infra.vm_types]
        return min(exec_times)

    def avg_communication_cost (self, node1, node2, job):
        if self.is_virtual_node(node1) or self.is_virtual_node(node2):
            return 0.0
        costs = []
        for t1 in self.infra.vm_types:
            for t2 in self.infra.vm_types:
                costs.append(self.pred.data_writing_time(node1[0], t1) + \
                   self.pred.data_reading_time(node1[0], node2[0], t2))
        return sum(costs)/len(costs)

    def compute_est_eft_lft(self, job, deadline):
        """
        Compute EST (Estimated Start Time) and EFT (Estimated Finish Time) for each task.
        This is the core of the IC-PCP algorithm.
        
        Returns:
            est: dict mapping each node to its EST value
            eft: dict mapping each node to its EFT value
        """
        est = {n: 0.0 for n in job.nodes}
        eft = {n: 0.0 for n in job.nodes}
        
        # Topological sort to process nodes in dependency order
        nodes = list(nx.topological_sort(job))
        
        for n in nodes:
            est[n] = 0.0
            for p in job.predecessors(n):
                est[n] = max(est[p] + self.avg_communication_cost(p, n, job) + self.min_computation_cost(p, job), est[n])

            # EFT = EST + min execution time
            eft[n] = est[n] + self.min_computation_cost(n, job)

        lft = {n: 0.0 for n in job.nodes}
        nodes = list(nx.topological_sort(job.reverse()))
        for n in nodes:
            lft[n] = deadline
            for c in job.successors(n):
                lft[n] = min(lft[n], lft[c] - self.min_computation_cost(c, job) - self.avg_communication_cost(n, c, job))
        
        return est, eft, lft
    


    def fix_schedule_with_colocation (self, job, eft, aft, sol):
        """
        We exploit co-location if possible to avoid some result writing,
        whose delay has been considered during scheduling execution.
        """
        tasklist = sorted(job.nodes, key=lambda n: eft[n], reverse=True)

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
                if not self.is_virtual_node(n) and not self.is_virtual_node(p) and sol.subtask2instance[p] != vm:
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

            # Virtual nodes have zero execution time
            if self.is_virtual_node(n):
                exec_time = 0.0
            else:
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
