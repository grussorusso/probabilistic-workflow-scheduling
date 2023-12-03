import math

#class VMType:
#    
#    def __init__ (self, name, cost, cores, provider, max_instances):
#        self.name = name
#        self.cost = cost
#        self.cores = cores
#        self.provider = provider
#
#    def __repr__ (self):
#        return self.name

class Infrastructure:

    def __init__ (self, vm_types):
        self.vm_types = vm_types
        self.providers = set()
        for vmt in vm_types:
            self.providers.add(vmt.provider)

    def get_cost_for_type (self, t):
        return self.vm_types[t].cost

    def all_instances (self):
        for vmt in self.vm_types:
            for i in range(vmt.max_instances):
                yield (vmt, i)

    def all_instances_of_type (self, vmt):
        for i in range(vmt.max_instances):
            yield (vmt, i)

    def __repr__ (self):
        return f"{self.vm_types}"
