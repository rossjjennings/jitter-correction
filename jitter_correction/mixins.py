import numpy as np
import h5py
import typing
from typing import Iterator, Self, TypeVar, Generic, Any, overload
from dataclasses import dataclass, fields
from collections.abc import Mapping
import importlib
import sys

def hdf5_save_item(key: str, item: Any, grp: h5py.Group):
    '''
    Save an item named `key`, which may be an Hdf5Serializable
    value, a mapping, an array-like object, or a numpy scalar,
    to the specified HDF5 Group.
    '''
    if isinstance(item, Hdf5Serializable):
        subgrp = grp.create_group(key)
        item.save_group(subgrp)
    elif isinstance(item, Mapping):
        subgrp = grp.create_group(key)
        for subkey, subitem in item.items():
            hdf5_save_item(subkey, subitem, subgrp)
    elif isinstance(item, list):
        subgrp = grp.create_group(key)
        for i, subitem in enumerate(item):
            hdf5_save_item(str(i), subitem, subgrp)
    else:
        # assume item is an ndarray or numpy scalar
        grp.create_dataset(key, data=item)

def hdf5_load_item(key: str, hint_type: Any, grp: h5py.Group):
    '''
    Load an item named `key`, which may be an Hdf5Serializable
    value, a mapping, an array-like object, or a numpy scalar,
    from the specified HDF5 Group, using type hints as a guide.
    '''
    type_conflict = (
        '_module' in grp.attrs
        and hasattr(hint_type, '__module__')
        and grp.attrs['_module'] != hint_type.__module__
    )
    type_conflict |= (
        '_type_qualname' in grp.attrs
        and hasattr(hint_type, '__qualname__')
        and grp.attrs['_type_qualname'] != hint_type.__qualname__
    )
    if type_conflict or isinstance(hint_type, TypeVar):
        module = importlib.import_module(grp.attrs['_module'])
        hint_type = getattr(module, grp.attrs['_type_qualname'])
    try:
        is_serializable = issubclass(hint_type, Hdf5Serializable)
    except TypeError:
        is_serializable = False
    if is_serializable:
        return hint_type.from_group(grp[key])
    elif (hasattr(hint_type, '__origin__')
          and issubclass(hint_type.__origin__, Mapping)):
        mapping = hint_type.__origin__()
        key_type, value_type = hint_type.__args__
        for subkey in grp[key]:
            mapping[subkey] = hdf5_load_item(subkey, value_type, grp[key])
        return mapping
    elif (hasattr(hint_type, '__origin__')
          and issubclass(hint_type.__origin__, list)):
        sequence = hint_type.__origin__()
        value_type, = hint_type.__args__
        i = 0
        while (subkey := str(i)) in grp[key]:
            sequence.append(hdf5_load_item(subkey, value_type, grp[key]))
            i += 1
        return sequence
    else:
        # assume item is an ndarray or numpy scalar
        return np.asarray(grp[key])[()]

@dataclass(slots=True)
class Hdf5Serializable:
    '''
    A mixin which allows class instances to be saved as HDF5 files.
    Subclasses are expected to be dataclasses containing only instance
    variables which are array-like or themselves Hdf5Serializable.
    '''
    def save_group(self, grp: h5py.Group):
        '''
        Save the data from this class instance to an HDF5 group.
        '''
        grp.attrs['_module'] = self.__class__.__module__
        grp.attrs['_type_qualname'] = self.__class__.__qualname__
        for field in fields(self):
            item = getattr(self, field.name)
            hdf5_save_item(field.name, item, grp)

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
        fields_dict = {}
        type_hints = typing.get_type_hints(cls)
        for field in fields(cls):
            fields_dict[field.name] = hdf5_load_item(
                field.name,
                type_hints[field.name],
                grp,
            )
        return cls(**fields_dict)

    @classmethod
    def from_hdf5(cls, filename: str):
        with h5py.File(filename, 'r') as f:
            instance = cls.from_group(f)
        return instance

@dataclass(slots=True)
class NpzSerializable:
    '''
    A mixin which allows class instances to be saved as NPZ files.
    Subclasses are expected to be dataclasses with only array-like
    instance variables.
    '''
    def save_npz(self, filename: str):
        '''
        Save the data from this class instance to an npz file.
        '''
        fields_dict = {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }
        np.savez(filename, **fields_dict)

    @classmethod
    def from_npz(cls, filename: str):
        '''
        Load data from an npz file and return an instance of this class.
        '''
        npz = np.load(filename)
        return cls(**npz)

@dataclass(slots=True)
class Serializable(Hdf5Serializable, NpzSerializable):
    pass

@dataclass(slots=True)
class RecordType(Serializable):
    '''
    A type which can be converted into a record stored in a Numpy record array.
    Subclasses must only have fields which are Numpy scalars or arrays.
    '''
    def __init_subclass__(cls, **kwargs):
        super(cls).__init_subclass__(**kwargs)
        for field in fields(cls):
            is_numpy_scalar = issubclass(field.type, np.generic)
            is_numpy_array = issubclass(field.type, np.ndarray)
            if not (is_numpy_scalar or is_numpy_array):
                raise ValueError(
                    f"Field {field.name} has type {field.type}, "
                    "which is neither a numpy scalar type nor an "
                    "ndarray subclass."
                )

    def __repr__(self):
        descr = f"{type(self).__name__}(\n"
        items = []
        for field in fields(self):
            value = getattr(self, field.name)
            if len(value.shape) > 0:
                with np.printoptions(linewidth=sys.maxsize, threshold=4, edgeitems=2):
                    value_str = repr(value)
            elif isinstance(value, np.floating):
                value_str = f"{value:.12g}"
            else:
                value_str = str(value)
            items.append(f"    {field.name}={value_str},")
        descr += "\n".join(items)
        descr += "\n)"
        return descr

    def as_record(self):
        dtype = []
        values = []
        for field in fields(self):
            value = getattr(self, field.name)
            values.append(value)
            if issubclass(field.type, np.ndarray):
                dtype.append((field.name, value.dtype, value.shape))
            else:
                dtype.append((field.name, type(value)))
        values = tuple(values)
        record = np.rec.fromrecords(values, dtype=dtype)[()]
        return record

R = TypeVar("R", covariant=True)

class RecordContainer(Generic[R]):
    '''
    Given a record type (class inheriting from NamedTuple), allows creating
    a container type which internally stores records of the given type in a
    record array, and allows iterating, slicing, and accessing fields by name.

    Creating the container type is done by indexing `RecordContainer` with
    the corresponding record type and subclassing the resulting mixin class
    (e.g., `RecordContainer[MyTuple]`).
    '''
    def __class_getitem__(self, record_type: type) -> type:
        '''
        Construct a mixin class representing a container for a record type
        '''
        @dataclass(slots=True)
        class RecordContainerAlias(Serializable):
            '''
            A mixin representing a container for a specific record type
            '''
            data: np.recarray

            def __init__(self, data: np.ndarray) -> None:
                '''
                Transform the input data into a record array
                '''
                self.data = np.rec.array(data)

            def __iter__(self) -> Iterator[R]:
                '''
                Iterate over the records stored in this container
                '''
                for rec in self.data:
                    yield record_type(*rec)

            @overload
            def __getitem__(self, key: int) -> Self:
                ...

            @overload
            def __getitem__(self, key: slice) -> R:
                ...

            def __getitem__(self, key: int | slice) -> R | Self:
                '''
                Allow slicing the array to return new container objects
                '''
                item = self.data[key]
                if item.shape == ():
                    return record_type(*item)
                else:
                    return type(self)(item)

            def __getattr__(self, attr: str) -> np.ndarray:
                '''
                Get fields as individual arrays
                '''
                return getattr(self.data, attr)

        return RecordContainerAlias

    def __getattr__(self, attr: str) -> np.ndarray:
        '''
        A type hack: without this, the type checker can't figure out that
        subclasses of `RecordContainerAlias` define `__getattr__()`.
        '''
        raise AttributeError(
            f"'RecordContainer' object has no attribute '{attr}'"
        )
