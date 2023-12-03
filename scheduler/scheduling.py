import networkx as nx
from functools import total_ordering
from scheduler.job import Operator, Job

@total_ordering
class ScheduleEntry:
    """
    A task scheduled for execution with an Estimated Starting (and Finish) Time.
    """

    def __init__ (self, task, est, eft=None):
        self.task = task
        self.est = est
        if eft is not None:
            self.eft = eft
        else:
            self.eft = est + 1.0

    def __lt__ (self, other):
        return self.est < other.est

    def __repr__ (self):
        return f"[{self.task}] {self.est}--{self.eft}"

class SchedulingSolution:
    """
    Solution computed by the scheduler.
    subtask2instance: maps each job node to a VM instance
    vm_schedule: maps each VM to its "schedule", i.e. a list of ScheduleEntry
    """

    def __init__(self, ignore_instance_numbers=False):
        self.subtask2instance = {}
        self.vm_schedule = {}
        self.ignore_instance_numbers = ignore_instance_numbers

    def __lt__ (self, other):
        # just to break ties..
        for vm in sorted(self.vm_schedule.keys()):
            for vm2 in sorted(other.vm_schedule.keys()):
                return vm[0].family < vm2[0].family

    def copy (self):
        cp = SchedulingSolution()
        cp.subtask2instance = self.subtask2instance.copy()
        cp.vm_schedule = {vm: self.vm_schedule[vm].copy() for vm in self.vm_schedule}

        return cp

    def sort_schedules (self):
        for vm in self.vm_schedule:
            self.vm_schedule[vm] = sorted(self.vm_schedule[vm])

    def add_scheduled_subtask (self, subtask, instance, est, eft):
        self.subtask2instance[subtask] = instance
        if not instance in self.vm_schedule:
            self.vm_schedule[instance] = []
        entry = ScheduleEntry(subtask, est, eft)
        self.vm_schedule[instance].append(entry)

    def is_feasible (self, infrastructure, deadline=None):
        """
        is_feasible checks whether vCPU limits are violated.
        """
        cpu_allocations = {p:[] for p in infrastructure.providers}

        for vm in self.vm_schedule:
            if len(self.vm_schedule[vm])==0:
                continue
            start_time = min([e.est for e in self.vm_schedule[vm]])
            end_time = max([e.eft for e in self.vm_schedule[vm]])
            if deadline is not None and end_time > deadline:
                return False

            cpu_allocations[vm[0].provider].append((start_time, vm[0].core_count))
            cpu_allocations[vm[0].provider].append((end_time, -vm[0].core_count))

        for provider in cpu_allocations:
            allocations = sorted(cpu_allocations[provider])
            total_cores = 0
            for t,c in allocations:
                total_cores += c
                if total_cores > provider.get_vcpu_limit():
                    return False

        return True

    def cost (self):
        cost = 0.0
        for vm in self.vm_schedule:
            if len(self.vm_schedule[vm])==0:
                continue
            start_time = min([e.est for e in self.vm_schedule[vm]])
            end_time = max([e.eft for e in self.vm_schedule[vm]])
            cost += vm[0].cost * (end_time - start_time)

        return cost

    # TODO: create optional param predictor:
    # if predictor is given we can re-evaluate EFT of each task (with possible
    # cold starts), instead of relying on those computed by the resolution alg.
    def makespan (self):
        t = 0.0
        for vm in self.vm_schedule:
            for entry in self.vm_schedule[vm]:
                t = max(t, entry.eft)
        return t

    def __repr__ (self):
        executions = []
        for vm in self.vm_schedule:
            executions.extend(map(lambda x: (vm, x.est, x.task, x.eft), self.vm_schedule[vm]))
        return str(executions)


    def plot (self):
        # Import libraries required only for plotting
        import plotly.express as px
        import pandas as pd
        import datetime

        entries=[]

        t0=datetime.datetime.now()

        for vm in self.vm_schedule:
            for e in self.vm_schedule[vm]:
                entries.append(dict(TaskIndex=e.task[1],Task=str(e.task), \
                        Start=pd.to_datetime(t0+datetime.timedelta(seconds=e.est)), \
                        Finish=pd.to_datetime(t0+datetime.timedelta(seconds=e.eft)),\
                        Resource=str(vm)))

        df = pd.DataFrame(entries)

        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", color="TaskIndex")
        fig.show()

    def to_execution_plan(self, job):
        return ExecutionPlan(job, self)


class ExecutionPlan (nx.DiGraph):
    """
    Higher-level representation of a scheduling solution as a "graph of (sub)graphs",
    where co-located communicating tasks are grouped into a subgraph.
    This is likely needed for distributed execution on top of SNAP (where we
    need to actually run whole graphs on each machine).
    """

    def __init__ (self, job, sched_solution):
        super().__init__(incoming_graph_data=None, name=f"ExecPlan_{job.name}")
        self.job = job
        self.vm2jobs, self.task2jobvm = self.__compute_subjobs(job, sched_solution)
        for vm in self.vm2jobs:
            for j in self.vm2jobs[vm]:
                self.add_node((j, vm))

        # Add intra-vm dependencies
        for vm in self.vm2jobs:
            for i in range(len(self.vm2jobs[vm])-1):
                self.add_edge((self.vm2jobs[vm][i],vm), (self.vm2jobs[vm][i+1], vm), label="Intra")

        self.tasks_with_remote_successors = set()

        # Add inter-vm dependencies
        for j,vm in self.nodes():
            for task in j.nodes():
                successors = job.successors(task)
                for s in successors:
                    succ_jobvm = self.task2jobvm[s]
                    if succ_jobvm != (j,vm):
                        self.tasks_with_remote_successors.add(task)
                        self.add_edge((j,vm), succ_jobvm)





    def __compute_subjobs (self, job, sched_solution):
        sol = sched_solution.copy()
        sol.sort_schedules()
        schedule = {vm: list(map(lambda x: x.task, sol.vm_schedule[vm])) for vm in sol.vm_schedule if len(sol.vm_schedule[vm]) > 0}

        vm2jobs = {}
        task2jobvm = {}

        for vm in schedule:
            jobs = []
            # Group scheduled task into partial jobs
            task_group = []
            for task in schedule[vm]:
                same_group = True

                # Check if the task belongs to the current group, if:
                # 1) all the task predecessors are in the group
                predecessors = job.predecessors(task) 
                for p in predecessors:
                    if not p in task_group:
                        same_group = False

                if len(task_group) == 0 or same_group:
                    task_group.append(task)
                else:
                    assert(len(task_group) > 0)
                    subjob = job.subgraph(task_group)
                    for t in task_group:
                        task2jobvm[t] = (subjob,vm)
                    jobs.append(subjob)
                    task_group = [task]

                # check if we need to terminate this group:
                # if any successor is on a different machine
                successors = job.successors(task)
                for s in successors:
                    if not s in schedule[vm]:
                        subjob = job.subgraph(task_group)
                        for t in task_group:
                            task2jobvm[t] = (subjob,vm)
                        jobs.append(subjob)
                        task_group = []
                        break

            if len(task_group) > 0:
                    subjob = job.subgraph(task_group)
                    for t in task_group:
                        task2jobvm[t] = (subjob,vm)
                    jobs.append(subjob)
            vm2jobs[vm] = jobs
        return vm2jobs, task2jobvm

