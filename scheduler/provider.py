#!/usr/bin/python3
import dataclasses as dc

from scheduler.vmprovider import VMProvider, VMType


@dc.dataclass
class FakeVMType(VMType):
    cost: float
    bandwidth_MBs: float

    def get_machine(self):
        pass

    def get_cost(self, time):
        super()
        return time * self.cost

    def __repr__ (self):
        return self.name

    def __hash__ (self):
        return hash(self.name)

    def __lt__ (self, vm):
        return self.cost < vm.cost


class FakeProvider(VMProvider):

    def __init__(self, config):
        super().__init__()

        self.__vcpu_limit = 576  # fallback in case the corresponding value is not found

    def get_vcpu_limit(self):
        return self.__vcpu_limit

    def set_vcpu_limit(self, limit):
        self.__vcpu_limit = limit

    @property
    def name(self):
        return "Fake"

    def load_vmtypes(self):
        pass

    def admissible_vm_instance_set(self, schedule, new_vm: VMType):
        vcpu_count = 0
        for vm_schedule in schedule:
            if vm_schedule.vmtype.provider.name == self.name:
                vcpu_count = vcpu_count + vm_schedule.vmtype.core_count
        if vcpu_count + new_vm.core_count < self.__vcpu_limit:
            return True
        return False

    # EMISILVE :: CHANGE MARKER
    def exceeded_vm_frame_schedule(self, schedule):
        vcpu_count = 0
        for vm_schedule in schedule:
            if vm_schedule.vmtype.provider.name == self.name:
                vcpu_count = vcpu_count + vm_schedule.vmtype.core_count
        return vcpu_count - self.__vcpu_limit
