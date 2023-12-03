from scheduler.job import *
from scheduler.infrastructure import *
from scheduler.scheduling import SchedulingSolution, ScheduleEntry

import heapq
import time

class TaskCompletion:
    def __init__ (self, task, vm):
        self.task = task
        self.vm = vm

    def __lt__ (self, other):
        return self.vm[0].family < other.vm[0].family

class VMBootCompleted:
    def __init__ (self, vm, first_task):
        self.vm = vm
        self.first_task = first_task

    def __lt__ (self, other):
        return self.vm[0].family < other.vm[0].family

class VMScheduledShutdown:
    def __init__ (self, vm):
        self.vm = vm

    def __lt__ (self, other):
        return self.vm[0].family < other.vm[0].family


def find_all_ready_tasks (job, sol, completed, running, blocked):
    ready = []
    for n in job.nodes():
        if n in completed or n in running or n in blocked:
            continue

        can_execute = True
        for p in job.predecessors(n):
            if not p in completed:
                # this subtask must wait
                can_execute = False
                break

        if can_execute:
            ready.append(n)
    return ready


def simulate_dyna (job, predictor, sol, task_durations, billing_period_sec=1):
    sol = sol.copy()
    actual_schedules = {}
    active_vm=set()
    blocked_tasks=[]

    events = [] 
    t = 0.0

    to_complete = set(list(job.nodes))
    completed = set()
    running = set()
    allocated_vcpus = {}
    busy_instances = {}
    idle_instances = {}
    vm_boot_time = {}
    vm_last_completion_time = {}

    task2vmtype = {}
    for vm,schedule in sol.vm_schedule.items():
        for entry in schedule:
            task = entry.task
            task2vmtype[task] = vm[0]

    def can_allocate_vm_type (vmt):
        provider = vmt.provider
        if allocated_vcpus.get(provider,0) + vmt.core_count > provider.get_vcpu_limit():
            return False
        if len(busy_instances.get(vmt, set())) + len(idle_instances.get(vmt,set())) >= vmt.max_instances:
            return False
        return True

    def schedule_task_completion (task, vm):
        data_reading_time = 0
        for p in job.predecessors(task):
            data_reading_time = max(data_reading_time, predictor.data_reading_time(p[0],task[0],vm[0]))

        completion_time = t + data_reading_time + task_durations[task]

        # check if we need to transfer output:
        # assume yes
        completion_time += predictor.data_writing_time(task[0], vm[0]) 

        # Update actual schedule
        if not vm in actual_schedules:
            actual_schedules[vm] = []
        entry = ScheduleEntry(task, t, completion_time)
        actual_schedules[vm].append(entry)
        
        running.add(task)
        heapq.heappush(events, (completion_time, TaskCompletion(task,vm)))


    # Schedule activation of required VMs for ready tasks
    for ready_task in find_all_ready_tasks(job, sol, completed, running, blocked_tasks):
        vm_type = task2vmtype[ready_task]
        if can_allocate_vm_type(vm_type):
            provider = vm_type.provider
            vm = (vm_type, time.time())
            allocated_vcpus[provider] = allocated_vcpus.get(provider, 0.0) + vm_type.core_count
            if not vm_type in busy_instances:
                busy_instances[vm_type] = set()
            busy_instances[vm_type].add(vm)
            running.add(ready_task)
            heapq.heappush(events, (t, VMBootCompleted(vm, ready_task)))
        else:
            if not ready_task in blocked_tasks:
                blocked_tasks.append(ready_task)

    while len(events) > 0:
        t, e = heapq.heappop(events)

        if isinstance(e, VMBootCompleted):
            vm = e.vm
            assert(allocated_vcpus[provider] <= vm[0].provider.get_vcpu_limit())
            schedule_task_completion(e.first_task, vm)
            vm_boot_time[vm] = t
        elif isinstance(e, TaskCompletion):
            task = e.task
            vm = e.vm
            to_complete.remove(task)
            running.remove(task)
            completed.add(task)
            busy_instances[vm[0]].remove(vm)
            if not vm[0] in idle_instances:
                idle_instances[vm[0]] = set()
            idle_instances[vm[0]].add(vm)
            vm_last_completion_time[vm] = t

            # schedule VM shut down at the end of the billing period
            shutdown_time = vm_boot_time[vm] + math.ceil(t/billing_period_sec)*billing_period_sec
            heapq.heappush(events, (shutdown_time, VMScheduledShutdown(vm)))
        elif isinstance(e, VMScheduledShutdown):
            # power off VM?
            vm = e.vm
            shutdown_time = vm_boot_time[vm] + math.ceil(vm_last_completion_time[vm]/billing_period_sec)*billing_period_sec
            if shutdown_time <= t and vm in idle_instances[vm[0]]:
                allocated_vcpus[provider] -= vm[0].core_count
                idle_instances[vm[0]].remove(vm)


        # schedule blocked and next tasks
        ready_tasks = blocked_tasks + find_all_ready_tasks(job, sol, completed, running, blocked_tasks)
        for ready_task in ready_tasks:
            assert(ready_task not in completed)
            vm_type = task2vmtype[ready_task]
            if len(idle_instances.get(vm_type, set())) > 0:
                vm = idle_instances[vm_type].pop()
                busy_instances[vm_type].add(vm)
                running.add(ready_task)
                schedule_task_completion(ready_task, vm)
                if ready_task in blocked_tasks:
                    blocked_tasks.remove(ready_task)
            else:
                if can_allocate_vm_type(vm_type):
                    provider = vm_type.provider
                    vm = (vm_type, time.time())
                    allocated_vcpus[provider] = allocated_vcpus.get(provider, 0.0) + vm_type.core_count
                    if not vm_type in busy_instances:
                        busy_instances[vm_type] = set()
                    busy_instances[vm_type].add(vm)
                    running.add(ready_task)
                    heapq.heappush(events, (t, VMBootCompleted(vm, ready_task)))
                    if ready_task in blocked_tasks:
                        blocked_tasks.remove(ready_task)
                else:
                    if not ready_task in blocked_tasks:
                        blocked_tasks.append(ready_task)


    if len(to_complete) > 0:
        print("Unfeasible!!!")
        return None, None 

    makespan = max(vm_last_completion_time.values())
    sol.vm_schedule = actual_schedules
    return makespan, sol
