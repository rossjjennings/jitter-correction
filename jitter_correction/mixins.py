import numpy as np
import h5py
import typing
from typing import Iterator, Self
from dataclasses import dataclass

class Hdf5Serializable:
    '''
    A mixin which allows class instances to be saved as HDF5 files.
    Subclasses are expected to be dataclasses with slots, with instance
    variables which are ArrayLike or themselves Hdf5Serializable.
    '''
    __slots__ = ()

    def save_group(self, grp: h5py.Group):
        '''
        Save the data from this class instance to an HDF5 group.
        '''
        for slot in self.__slots__:
            item = getattr(self, slot)
            if isinstance(item, Hdf5Serializable):
                subgrp = grp.create_group(slot)
                item.save_group(subgrp)
            else:
                # assume item is an ndarray or numpy scalar
                grp.create_dataset(slot, data=item)

    def save_hdf5(self, filename: str):
        '''
        Save the data from this class instance to an HDF5 file.
        '''
        with h5py.File(filename, 'w') as f:
            self.save_group(f)

    @classmethod
    def from_group(cls, grp: h5py.Group):
        '''
        Load data from an HDF5 group and return an instance of this class.
        '''
        slots_dict = {}
        type_hints = typing.get_type_hints(cls)
        for slot in cls.__slots__:
            try:
                recursive = issubclass(type_hints[slot], Hdf5Serializable)
            except TypeError:
                recursive = False
            if recursive:
                item = type_hints[slot].from_group(grp[slot])
                slots_dict[slot] = item
            else:
                # assume item is an ndarray or numpy scalar
                slots_dict[slot] = np.asarray(grp[slot])[()]
        return cls(**slots_dict)

    @classmethod
    def from_hdf5(cls, filename: str):
        with h5py.File(filename, 'r') as f:
            instance = cls.from_group(f)
        return instance

class NpzSerializable:
    '''
    A mixin which allows class instances to be saved as NPZ files.
    Subclasses are expected to be dataclasses with slots, with only
    ArrayLike instance variables.
    '''
    __slots__ = ()

    def save_npz(self, filename: str):
        '''
        Save the data from this class instance to an npz file.
        '''
        slots_dict = {slot: getattr(self, slot) for slot in self.__slots__}
        np.savez(filename, **slots_dict)

    @classmethod
    def from_npz(cls, filename: str):
        '''
        Load data from an npz file and return an instance of this class.
        '''
        npz = np.load(filename)
        return cls(**npz)

class RecordContainer(type):
    '''
    Given a record type (class inheriting from NamedTuple), allows creating
    a container type which internally stores records of the given type in a
    record array, and allows iterating, slicing, and accessing fields by name.

    Creating the container type is done by subclassing `RecordContainer` and
    giving the subclass an attribute `record_type` which stores the associated
    record type.
    '''
    def __new__(mcls, name, bases, namespace):
        try:
            record_type = namespace['record_type']
        except KeyError:
            raise TypeError("RecordContainer must have a record type")

        def __init__(self, data: np.ndarray):
            self.data = np.rec.array(data)

        def __iter__(self) -> Iterator[record_type]:
            for rec in self.data:
                yield record_type(*rec)

        def __getitem__(self, key) -> record_type | Self:
            item = self.data[key]
            if item.shape == ():
                return record_type(*item)
            else:
                return type(self)(item)

        def __getattr__(self, attr):
            return getattr(self.data, attr)

        namespace.update({
            '__init__': __init__,
            '__iter__': __iter__,
            '__getitem__': __getitem__,
            '__getattr__': __getattr__,
        })

        return super().__new__(mcls, name, bases, namespace)
