#!/usr/bin/python3

from abc import ABC, abstractmethod
from typing import List

import dataclasses as dc


_vmtypes = {}


@dc.dataclass(unsafe_hash=True)
class VMType(ABC):
    """An abstract class used to represent a virtual machine offering from a cloud provider"""
    name: str
    family: str
    provider: "VMProvider"
    core_count: int
    #core_frequency: float
    gpu_count: int
    memory_size: float
    details: str

    def __post_init__(self):
        _vmtypes[self.name] = self
        self.max_instances = min(2*int(self.provider.get_vcpu_limit()/self.core_count), 20)

    @staticmethod
    def get_by_name(name):
        return _vmtypes.get(name, None)

    @staticmethod
    def get_all():
        return tuple(_vmtypes.values())

    @abstractmethod
    def get_cost(self, time) -> float:
        pass

    def __str__(self):
        return "virtual machine type " + self.name + " provided by " + str(self.provider)

    def filter_by_compatibility(self, computations):
        result = []
        for computation in computations:
            if (self.gpu_count > 0 and computation.algorithm.gpu_bound == "yes") \
                or (self.gpu_count == 0 and computation.algorithm.gpu_bound == "no") \
                    or (computation.algorithm.gpu_bound != "yes" and computation.algorithm.gpu_bound != "no"):
                result.append(computation)
        return result



class VMProvider(ABC):

    @abstractmethod
    def load_vmtypes(self) -> List[VMType]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def admissible_vm_instance_set(self, schedule, new_vm) -> bool:
        pass

    # EMISILVE :: CHANGE MARKER
    @abstractmethod
    def exceeded_vm_frame_schedule(self, schedule) -> int:
        pass

    def __str__(self):
        return "provider " + self.name
