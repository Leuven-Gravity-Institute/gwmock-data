"""HDF5 files as a tree of groups and datasets, readable and writable without h5py.

:func:`write` and :func:`read` work on :class:`Group` and :class:`Dataset`. h5py is used when it is installed
(the ``hdf5`` extra); without it, files are written by a small built-in writer and read with pyfive, a pure-Python
HDF5 reader. Either way the files are ordinary HDF5: h5py, GWpy and the HDF5 tools read what the built-in writer
writes, and the built-in path reads what they write.

The built-in writer covers what the gwmock packages store: datasets of booleans, integers, 32- and 64-bit floats,
fixed-length strings and compound rows of those; attributes of numbers, arrays and strings; nested groups. It
writes data uncompressed; files h5py wrote with gzip compression read either way.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from gwmock_data.hdf5._writer import encode

__all__ = ["Dataset", "Group", "read", "write"]

Backend = Literal["auto", "h5py", "builtin"]


@dataclass
class Dataset:
    """An HDF5 dataset: an array and its attributes. A ``str`` is stored as a fixed-length UTF-8 string."""

    data: Any
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Group:
    """An HDF5 group: named members (groups or datasets) and attributes. The file itself is the root group."""

    members: dict[str, Group | Dataset] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)


def _h5py() -> Any | None:
    """Return h5py if it is installed, ``None`` if it is not; re-raise a broken installation."""
    try:
        return importlib.import_module("h5py")
    except ModuleNotFoundError as exc:
        if exc.name != "h5py":
            raise
        return None


def _array(data: Any) -> np.ndarray:
    """Return dataset or attribute data as a NumPy array, a ``str`` as UTF-8 bytes."""
    if isinstance(data, str):
        return np.asarray(np.bytes_(data.encode("utf-8")))
    return np.asarray(data)


def write(root: Group, path: str | Path, *, backend: Backend = "auto", compression: str | None = None) -> None:
    """Write ``root`` to ``path`` as an HDF5 file, replacing it.

    Args:
        root: The file's contents.
        path: Output file.
        backend: ``"h5py"``, ``"builtin"``, or ``"auto"`` for h5py when it is installed and the built-in writer
            otherwise.
        compression: A compression filter for h5py to apply to non-scalar datasets, such as ``"gzip"``. The
            built-in writer does not compress and ignores it.

    Raises:
        ModuleNotFoundError: If ``backend="h5py"`` and h5py is not installed.
        TypeError: If a dtype cannot be written.
    """
    # Before choosing a backend, so that the same tree means the same file whichever is installed.
    normalised = _normalised(root)
    h5py = _h5py() if backend in ("auto", "h5py") else None
    if backend == "h5py" and h5py is None:
        raise ModuleNotFoundError(
            "The h5py backend needs h5py, which is not installed. Install it with: pip install 'gwmock-data[hdf5]'",
            name="h5py",
        )
    path = Path(path)
    if h5py is None:
        path.write_bytes(encode(normalised))
        return
    with h5py.File(path, "w") as handle:
        _write_h5py(handle, root, compression)


def _normalised(group: Group) -> Group:
    """``group`` with every dataset's data and every attribute as an array.

    Raises:
        ValueError: If a member name is empty, ``.`` or ``..``, or contains ``/``, which HDF5 reads as a path.
    """
    members: dict[str, Group | Dataset] = {}
    for name, member in group.members.items():
        if not name or "/" in name or name in (".", ".."):
            raise ValueError(f"{name!r} cannot name an HDF5 group member: it must be non-empty and contain no '/'.")
        if isinstance(member, Group):
            members[name] = _normalised(member)
        else:
            members[name] = Dataset(_array(member.data), {key: _array(value) for key, value in member.attrs.items()})
    return Group(members, {key: _array(value) for key, value in group.attrs.items()})


def _write_h5py(handle: Any, group: Group, compression: str | None) -> None:
    """Write ``group``'s attributes and members into an open h5py group.

    Strings are handed to h5py as ``str``, which it stores as variable-length UTF-8 strings, as GWpy does; the
    built-in writer stores fixed-length ones. Both read back as ``str``.
    """
    for key, value in group.attrs.items():
        handle.attrs[key] = value if isinstance(value, str) else _array(value)
    for name, member in group.members.items():
        if isinstance(member, Group):
            _write_h5py(handle.create_group(name), member, compression)
            continue
        if isinstance(member.data, str):
            dataset = handle.create_dataset(name, data=member.data)
        else:
            data = _array(member.data)
            dataset = handle.create_dataset(name, data=data, compression=compression if data.ndim else None)
        for key, value in member.attrs.items():
            dataset.attrs[key] = value if isinstance(value, str) else _array(value)


def read(path: str | Path, *, backend: Backend = "auto") -> Group:
    """Read the HDF5 file at ``path`` into a :class:`Group`.

    Datasets are read into NumPy arrays (a scalar as a 0-d array). A scalar string, as a dataset or an
    attribute, is returned as ``str``; an array of fixed-length strings stays a NumPy array of bytes.

    Args:
        path: Input file.
        backend: ``"h5py"``, ``"builtin"`` (pyfive), or ``"auto"`` for h5py when it is installed and pyfive
            otherwise.

    Raises:
        ModuleNotFoundError: If ``backend="h5py"`` and h5py is not installed.
    """
    h5py = _h5py() if backend in ("auto", "h5py") else None
    if backend == "h5py" and h5py is None:
        raise ModuleNotFoundError(
            "The h5py backend needs h5py, which is not installed. Install it with: pip install 'gwmock-data[hdf5]'",
            name="h5py",
        )
    if h5py is not None:
        with h5py.File(Path(path), "r") as handle:
            return _read_tree(handle, lambda item: isinstance(item, h5py.Group))
    pyfive = importlib.import_module("pyfive")
    with pyfive.File(str(path)) as handle:
        return _read_tree(handle, lambda item: isinstance(item, pyfive.Group))


def _read_tree(handle: Any, is_group: Any) -> Group:
    """Copy an open h5py or pyfive group, recursively, into a :class:`Group`."""
    members: dict[str, Group | Dataset] = {}
    for name in handle:
        item = handle[name]
        if is_group(item):
            members[name] = _read_tree(item, is_group)
        else:
            members[name] = Dataset(_value(item[()]), {key: _value(value) for key, value in item.attrs.items()})
    return Group(members, {key: _value(value) for key, value in handle.attrs.items()})


def _value(value: Any) -> Any:
    """A read value as returned: bytes and NumPy strings as ``str``, everything else as a NumPy array."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, str):
        return value
    return np.asarray(value)
