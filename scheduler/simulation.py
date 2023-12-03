from scheduler.job import *
from scheduler.infrastructure import *
from scheduler.scheduling import SchedulingSolution, ScheduleEntry

import heapq

class TaskCompletion:
    def __init__ (self, task, vm):
        self.task = task
        self.vm = vm

    def __lt__ (self, other):
        return self.vm[0].family < other.vm[0].family

class VMBootCompleted:
    def __init__ (self, vm):
        self.vm = vm

    def __lt__ (self, other):
        return self.vm[0].family < other.vm[0].family


def find_all_ready_tasks (job, sol, completed):
    for vm in sol.vm_schedule:
        sched = sol.vm_schedule[vm]
        next_task = None

        # find first non completed task on VM
        for entry in sched:
            if not entry.task in completed:
                next_task = entry.task
                break
        if next_task is None:
            continue

        can_execute = True
        for p in job.predecessors(next_task):
            if not p in completed:
                # this subtask must wait
                can_execute = False
                break

        if can_execute:
            yield next_task

def get_vm_ready_task (job, sol, completed, running, vm):
    sched = sol.vm_schedule[vm]
    next_task = None

    # find first non completed task on VM
    for entry in sched:
        if not entry.task in completed and not entry.task in running:
            next_task = entry.task
            break
    if next_task is None:
        return None

    can_execute = True
    for p in job.predecessors(next_task):
        if not p in completed:
            return None

    return next_task

def simulate (job, predictor, sol, task_durations):
    sol = sol.copy()
    actual_schedules = {}
    active_vm=set()
    blocked_vm=[]

    events = [] 
    t = 0.0

    to_complete = set(list(job.nodes))
    completed = set()
    running = set()
    allocated_vcpus = {}
    allocated_instances = {}

    # Schedule activation of required VMs for ready tasks
    for ready_task in find_all_ready_tasks(job, sol, completed):
        vm = sol.subtask2instance[ready_task]
        provider = vm[0].provider
        if allocated_vcpus.get(provider,0) + vm[0].core_count <= provider.get_vcpu_limit() and allocated_instances.get(vm[0], 0) < vm[0].max_instances:
            allocated_vcpus[provider] = allocated_vcpus.get(provider, 0.0) + vm[0].core_count
            allocated_instances[vm[0]] = allocated_instances.get(vm[0], 0) + 1
            heapq.heappush(events, (t, VMBootCompleted(vm)))
        else:
            blocked_vm.append(vm)


    while len(events) > 0:
        t, e = heapq.heappop(events)

        if isinstance(e, VMBootCompleted):
            vm = e.vm
            provider = vm[0].provider
            assert(allocated_vcpus[provider] <= provider.get_vcpu_limit())
            #print(f"{provider}: {allocated_vcpus[provider]}")
            active_vm.add(vm)

        elif isinstance(e, TaskCompletion):
            task = e.task
            vm = e.vm
            provider = vm[0].provider
            to_complete.remove(task)
            running.remove(task)
            completed.add(task)
            sol.vm_schedule[vm] = sol.vm_schedule[vm][1:]

            # power off VM?
            if len(sol.vm_schedule[vm]) == 0:
                active_vm.remove(vm)
                allocated_vcpus[provider] -= vm[0].core_count
                allocated_instances[vm[0]] -= 1
                started = 0
                for bvm in blocked_vm:
                    provider = bvm[0].provider
                    if allocated_vcpus.get(provider,0) + bvm[0].core_count <= provider.get_vcpu_limit() and\
                            allocated_instances.get(bvm[0], 0) < bvm[0].max_instances:
                        allocated_vcpus[provider] += bvm[0].core_count
                        allocated_instances[bvm[0]] = allocated_instances.get(bvm[0],0) + 1
                        heapq.heappush(events, (t, VMBootCompleted(bvm)))
                        started += 1
                    else:
                        break
                        # TODO: should we power on the next blocked VM if possible?
                blocked_vm = blocked_vm[started:]

            # schedule powering on of VMs for successors
            for p in job.successors(task):
                provider = vm[0].provider
                vm = sol.subtask2instance[p]
                is_first = sol.vm_schedule[vm][0].task == p
                if is_first and not vm in active_vm:
                        if allocated_vcpus.get(provider,0) + vm[0].core_count <= provider.get_vcpu_limit() and\
                            allocated_instances.get(vm[0], 0) < vm[0].max_instances:
                            allocated_vcpus[provider] += vm[0].core_count
                            allocated_instances[vm[0]] =allocated_instances.get(vm[0], 0) + 1
                            heapq.heappush(events, (t, VMBootCompleted(vm)))
                        else:
                            blocked_vm.append(vm)

        for vm in active_vm:
            task = get_vm_ready_task(job, sol, completed, running, vm)
            if task is None:
                continue

            data_reading_time = 0
            for p in job.predecessors(task):
                if sol.subtask2instance[p] != vm:
                    data_reading_time = max(data_reading_time, predictor.data_reading_time(p[0],task[0],vm[0]))

            completion_time = t + data_reading_time + task_durations[task]

            # check if we need to transfer output
            colocated_successors = True
            for p in job.successors(task):
                if sol.subtask2instance[p] != vm:
                    colocated_successors = False
            if not colocated_successors:
                    completion_time += predictor.data_writing_time(task[0], vm[0]) 

            # Update actual schedule
            if not vm in actual_schedules:
                actual_schedules[vm] = []
            entry = ScheduleEntry(task, t, completion_time)
            actual_schedules[vm].append(entry)
            
            running.add(task)
            heapq.heappush(events, (completion_time, TaskCompletion(task,vm)))

    if len(to_complete) > 0:
        print("Unfeasible!!!")
        return None, None 

    makespan = t
    sol.vm_schedule = actual_schedules
    return makespan, sol
