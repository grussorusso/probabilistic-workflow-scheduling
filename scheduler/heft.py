from scheduler.job import *
from scheduler.infrastructure import *
from scheduler.scheduling import SchedulingSolution, ScheduleEntry

class HEFT:

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

                if min_eft is None or eft < min_eft:
                    min_est = est
                    min_eft = eft
                    min_eft_vm = vm

            aft[n] = min_eft
            print(f"Scheduling {n} to {min_eft_vm} starting at {min_est}")
            self.sol.add_scheduled_subtask(n, min_eft_vm, min_est, min_eft)

        self.fix_schedule_with_colocation(job, rankU, aft, self.sol)
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
        exec_times = [self.pred.exec_time(node[0], job, vmt) for vmt in self.infra.vm_types]
        return sum(exec_times)/len(exec_times)

    def avg_communication_cost (self, node1, node2, job):
        costs = []
        for t1 in self.infra.vm_types:
            for t2 in self.infra.vm_types:
                costs.append(self.pred.data_writing_time(node1[0], t1) + \
                   self.pred.data_reading_time(node1[0], node2[0], t2))
        return sum(costs)/len(costs)

    def compute_rankU (self, job):
        nodes = list(reversed(list(nx.topological_sort(job))))
        ru = {n: 0.0 for n in list(job.nodes)}
        while True:
            delta = 0
            for n in nodes:
                succ_rank = [self.avg_communication_cost(n,succ, job) + ru[succ] for succ in job.successors(n)]
                if len(succ_rank) == 0:
                    succ_rank = [0.0]
                old_value = ru[n]
                ru[n] = max(ru[n], self.avg_computation_cost(n, job) + max(succ_rank))
                delta = max(delta, ru[n]-old_value)
            if delta < 0.0001:
                break
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



class MOHEFTCandidateSol:

    def __init__ (self, base_sol=None):
        if base_sol is None:
            self.aft = {}
            self.sol = SchedulingSolution()
        else:
            self.aft = base_sol.aft.copy()
            self.sol = base_sol.sol.copy()




class MOHEFT (HEFT):

    def __init__ (self, infra, pred, K=3):
        super().__init__(infra,pred)
        self.K = K
        self.S = None

    def schedule (self, job, deadline):
        assert(isinstance(job, Job))
        assert(deadline > 0.0)

        if self.S is None:
            self.S = []
            for k in range(self.K):
                self.S.append(MOHEFTCandidateSol())

        I = list(self.infra.all_instances())

        rankU = self.compute_rankU(job)
        tasklist = sorted(job.nodes, key=lambda n: rankU[n], reverse=True)

        while len(tasklist) > 0:
            n = tasklist[0]
            tasklist = tasklist[1:]

            S1 = [] # S'

            # Scheduling n on each vm
            for vm in I:
                #for k in range(self.K):
                for k in range(len(self.S)): # may be less than K
                    # Compute EST based on predecessors (Eq. 5, HEFT)
                    est = 0.0
                    data_reading_time = 0.0
                    for p in job.predecessors(n):
                        est = max(est, self.S[k].aft[p])
                        if self.S[k].sol.subtask2instance[p] != vm:
                            data_reading_time = max(data_reading_time, self.pred.data_reading_time(p[0],n[0],vm[0]))

                    # Compute EST based on availability (Eq. 5, HEFT)
                    # Insertion-based policy
                    if not vm in self.S[k].sol.vm_schedule:
                        self.S[k].sol.vm_schedule[vm] = []
                    est, vm_first = self.min_schedulable_time (n, vm, self.S[k].sol.vm_schedule[vm], job, est)
                    graph_first = len(list(job.predecessors(n)))==0

                    # EFT (Eq. 6, HEFT)
                    exec_time = self.pred.exec_time(n[0], job, vm[0], first_on_the_machine=vm_first,
                            first_in_the_graph=graph_first) +\
                        data_reading_time +\
                        self.pred.data_writing_time(n[0], vm[0]) # We are assuming that successors are not co-located here...
                    eft = est + exec_time
                    new_sol = MOHEFTCandidateSol(self.S[k])
                    new_sol.sol.add_scheduled_subtask(n, vm, est, eft)
                    new_sol.aft[n] = eft

                    if new_sol.sol.is_feasible(self.infra, deadline):
                        S1.append(new_sol)

            # Choose best K solutions
            self.S = self.choose_solutions(S1) 

        returned_sol = self.finalize_solution(job, rankU, deadline)
        return returned_sol

    def finalize_solution(self, job, rankU, deadline):
        for s in self.S:
            self.fix_schedule_with_colocation(job, rankU, s.aft, s.sol)

        # Pick the cheapest sol subject to deadline
        objC, objT = self.evaluate_solutions(self.S)
        candidates = sorted(self.S, key=lambda x: objC[x])

        #for sol in candidates:
        #    print(f"Candidate sol: cost={objC[sol]} and RT {objT[sol]}")

        for sol in candidates:
            if objT[sol] <= deadline:
                #print(f"Chosen sol: cost={objC[sol]} and RT {objT[sol]}")
                return sol.sol

        if len(candidates) == 0:
            return None
        
        # no feasible solution...
        print("[WARNING] no sol meets the deadline")
        return candidates[-1].sol 

    def evaluate_solutions (self,solutions):
        objC = {s: s.sol.cost() for s in solutions}
        objT = {s: s.sol.makespan() for s in solutions}
        return (objC, objT)


    def choose_solutions (self, solutions):
        if len(solutions) < 1:
            return solutions
        
        objC, objT = self.evaluate_solutions(solutions)

        Cmax = max(objC.values())
        Cmin = min(objC.values())
        if Cmax == Cmin:
            Cmax = Cmin + 0.0001
        Tmax = max(objT.values())
        Tmin = min(objT.values())
        if Tmax == Tmin:
            Tmax = Tmin + 0.0001

        Sp = {s: [] for s in solutions}
        np = {s: 0 for s in solutions}
        rank = {}
        fronts = []
        fronts.append([])

        for p in solutions:
            for q in solutions:
                if objT[p] < objT[q] and objC[p] < objC[q]:
                    Sp[p].append(q)
                elif objT[p] > objT[q] and objC[p] > objC[q]:
                    np[p] += 1

            if np[p] == 0:
                rank[p] = 1 # p belongs to Front 1 
                fronts[0].append(p)

        i=0
        while len(fronts[i]) > 0:
            Q = []
            for p in fronts[i]:
                for q in Sp[p]:
                    np[q] -= 1

                    if np[q] == 0:
                        rank[q] = i+1
                        Q.append(q)
            fronts.append(Q)
            i = i + 1
        fronts = fronts[:-1]

        chosen = []

        # Crowding distance
        for f in fronts:
            d={p: 0.0 for p in f}
            f.sort(key=lambda p: objC[p])
            d[f[0]] = float("inf")
            d[f[-1]] = float("inf")

            for k in range(1,len(f)-1):
                d[f[k]] += (objC[f[k+1]] - objC[f[k-1]])/(Cmax-Cmin)

            f.sort(key=lambda p: objT[p])
            d[f[0]] = float("inf")
            d[f[-1]] = float("inf")

            for k in range(1,len(f)-1):
                d[f[k]] += (objT[f[k+1]] - objT[f[k-1]])/(Tmax-Tmin)

            f.sort(reverse=True, key=lambda x : d[x])
            chosen.extend(f)
            if len(chosen) >= self.K:
                break
        
        if len(chosen) < self.K:
            return chosen
        else:
            return chosen[:self.K]


class CloudMOHEFT (MOHEFT):
    """
    Smarter MOHEFT for on-demand VMs.
    """

    def __init__ (self, infra, pred, K=3, enforce_deadline_on_partial_solutions=True, verbose=False):
        super().__init__(infra,pred, K)
        self.enforce_deadline_on_partial_solutions = enforce_deadline_on_partial_solutions
        self.verbose = verbose

    def schedule (self, job, deadline, return_frontier=False):
        assert(isinstance(job, Job))
        assert(deadline > 0.0)

    

        if self.S is None:
            self.S = []
            for k in range(self.K):
                self.S.append(MOHEFTCandidateSol())

        I = list(self.infra.all_instances())


        rankU = self.compute_rankU(job)
        tasklist = sorted(job.nodes, key=lambda n: rankU[n], reverse=True)

        while len(tasklist) > 0:
            if self.verbose:
                print(f"{len(tasklist)} tasks remaining.")
            n = tasklist[0]
            tasklist = tasklist[1:]

            S1 = [] # S'

            # Scheduling n on each vm
            #for k in range(self.K):
            for k in range(len(self.S)): # may be less than K
                _I = set() # reduced set of instances

                # Reuse all instances already in use
                for vm in self.S[k].sol.vm_schedule:
                    _I.add(vm)

                # Add a new instance of any type (if possible)
                for vmt in self.infra.vm_types:
                    for vm in self.infra.all_instances_of_type(vmt):
                        if not vm in _I:
                            _I.add(vm)
                            break

                for vm in _I:
                    # Compute EST based on predecessors (Eq. 5, HEFT)
                    est = 0.0
                    data_reading_time = 0.0
                    for p in job.predecessors(n):
                        est = max(est, self.S[k].aft[p])
                        if self.S[k].sol.subtask2instance[p] != vm:
                            data_reading_time = max(data_reading_time, self.pred.data_reading_time(p[0],n[0],vm[0]))

                    # Compute EST based on availability (Eq. 5, HEFT)
                    # Insertion-based policy
                    if not vm in self.S[k].sol.vm_schedule:
                        self.S[k].sol.vm_schedule[vm] = []
                    est, vm_first = self.min_schedulable_time (n, vm, self.S[k].sol.vm_schedule[vm], job, est)
                    graph_first = len(list(job.predecessors(n)))==0

                    # EFT (Eq. 6, HEFT)
                    exec_time = self.pred.exec_time(n[0], job, vm[0], first_in_the_graph=graph_first,
                            first_on_the_machine=vm_first) + data_reading_time +\
                        self.pred.data_writing_time(n[0], vm[0]) # We are assuming that successors are not co-located here...
                    eft = est + exec_time
                    new_sol = MOHEFTCandidateSol(self.S[k])
                    new_sol.sol.add_scheduled_subtask(n, vm, est, eft)
                    new_sol.aft[n] = eft

                    if not self.enforce_deadline_on_partial_solutions or new_sol.sol.is_feasible(self.infra, deadline):
                        S1.append(new_sol)

            # Choose best K solutions
            self.S = self.choose_solutions(S1) 
            if len(self.S) < 1:
                print("Unfeasible!!!")
                return None

        final_sol = self.finalize_solution(job, rankU, deadline)
        print(f"CloudMOHEFT makespan found: {final_sol.makespan()}")

        if return_frontier:
            return final_sol, [s.sol for s in self.S]
        return final_sol
