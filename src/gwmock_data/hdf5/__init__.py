"""HDF5 files as a tree of groups and datasets, readable and writable without h5py.

:func:`write` and :func:`read` work on :class:`Group` and :class:`Dataset`; :func:`update_attrs` changes the
attributes of an existing file. h5py is used when it is installed
(the ``hdf5`` extra); without it, files are written by a small built-in writer and read with pyfive, a pure-Python
HDF5 reader. Either way the files are ordinary HDF5: h5py, GWpy and the HDF5 tools read what the built-in writer
writes, and the built-in path reads what they write.

The built-in writer covers what the gwmock packages store: datasets of booleans, integers, 32- and 64-bit floats,
fixed-length strings and compound rows of those; attributes of numbers, arrays and strings, a string of any
length (stored, as h5py stores it, as a variable-length string), a number or array up to 64 KiB; nested groups.
It writes data uncompressed; files h5py wrote with gzip compression read either way.
"""

from __future__ import annotations

import importlib
import shutil
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from gwmock_data.hdf5._writer import encode

__all__ = ["Dataset", "Group", "read", "update_attrs", "write"]

Backend = Literal["auto", "h5py", "builtin"]


@dataclass
class Dataset:
    """An HDF5 dataset: an array and its attributes. A ``str`` is stored as a fixed-length UTF-8 string.

    ``data`` is ``None`` for a dataset read with ``read(..., data=False)``.
    """

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
        ValueError: If a member name cannot name an HDF5 member, or a string attribute contains a NUL character.
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


def _check_name(name: str) -> None:
    """Refuse a member name that is empty, ``.`` or ``..``, or contains ``/``, which HDF5 reads as a path."""
    if not name or "/" in name or name in (".", ".."):
        raise ValueError(f"{name!r} cannot name an HDF5 group member: it must be non-empty and contain no '/'.")


def _normalised(group: Group) -> Group:
    """``group`` with every dataset's data and every attribute as an array.

    Raises:
        ValueError: If a member name is empty, ``.`` or ``..``, or contains ``/``, which HDF5 reads as a path.
    """
    members: dict[str, Group | Dataset] = {}
    for name, member in group.members.items():
        _check_name(name)
        if isinstance(member, Group):
            members[name] = _normalised(member)
        else:
            members[name] = Dataset(_array(member.data), _attribute_arrays(member.attrs))
    return Group(members, _attribute_arrays(group.attrs))


def _attribute_arrays(attrs: dict[str, Any]) -> dict[str, Any]:
    """Attributes as arrays, a ``str`` kept as it is: it is written as a variable-length string."""
    return {key: _text(key, value) if isinstance(value, str) else _array(value) for key, value in attrs.items()}


def _text(key: str, value: str) -> str:
    """A string attribute, refused if it holds a NUL.

    It is stored as a null-terminated variable-length string, as h5py stores one: h5py refuses to write a NUL into
    one and stops reading at the first, so a string holding one would not read back the same everywhere.

    Raises:
        ValueError: If ``value`` contains a NUL character.
    """
    if "\0" in value:
        raise ValueError(f"String attribute {key!r} contains a NUL character, which an HDF5 string cannot hold.")
    return value


def _write_h5py(handle: Any, group: Group, compression: str | None) -> None:
    """Write ``group``'s attributes and members into an open h5py group.

    Strings are handed to h5py as ``str``, which it stores as variable-length UTF-8 strings, as GWpy does. The
    built-in writer does the same for attributes and stores a string dataset fixed-length; all read back as
    ``str``.
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


def read(
    path: str | Path, *, backend: Backend = "auto", members: Iterable[str] | None = None, data: bool = True
) -> Group:
    """Read the HDF5 file at ``path`` into a :class:`Group`.

    Datasets are read into NumPy arrays (a scalar as a 0-d array). A scalar string, as a dataset or an
    attribute, is returned as ``str``; an array of fixed-length strings stays a NumPy array of bytes.

    Args:
        path: Input file.
        backend: ``"h5py"``, ``"builtin"`` (pyfive), or ``"auto"`` for h5py when it is installed and pyfive
            otherwise.
        members: Names of the top-level members to read; the others are not read at all, which matters for a
            large file of which only a few columns are wanted. A name the file lacks is absent from the
            result. ``None``, the default, reads every member. The file's own attributes are always read.
        data: ``False`` reads the structure and the attributes only: every :class:`Dataset` comes back with
            ``data=None`` and its samples are never read, which matters for inspecting the metadata of a
            large file.

    Raises:
        ModuleNotFoundError: If ``backend="h5py"`` and h5py is not installed.
        TypeError: If ``members`` is a single string rather than a collection of names.
        ValueError: If a name in ``members`` is empty, ``.`` or ``..``, or contains ``/``: only top-level members
            can be selected, not paths into groups.
    """
    if isinstance(members, (str, bytes)):
        # A string is iterable, so it would be read as one-character names and silently select nothing.
        raise TypeError(f"members must be a collection of names, such as [{members!r}]; got a single string.")
    if members is not None:
        members = list(members)
        for name in members:
            _check_name(name)
    h5py = _h5py() if backend in ("auto", "h5py") else None
    if backend == "h5py" and h5py is None:
        raise ModuleNotFoundError(
            "The h5py backend needs h5py, which is not installed. Install it with: pip install 'gwmock-data[hdf5]'",
            name="h5py",
        )
    if h5py is not None:
        with h5py.File(Path(path), "r") as handle:
            return _read_tree(handle, lambda item: isinstance(item, h5py.Group), members, data=data)
    pyfive = importlib.import_module("pyfive")
    with pyfive.File(str(path)) as handle:
        return _read_tree(handle, lambda item: isinstance(item, pyfive.Group), members, data=data)


def _read_tree(handle: Any, is_group: Any, only: Iterable[str] | None = None, *, data: bool = True) -> Group:
    """Copy an open h5py or pyfive group, recursively, into a :class:`Group`; at the top, only ``only``."""
    names = list(handle) if only is None else [name for name in dict.fromkeys(only) if name in handle]
    members: dict[str, Group | Dataset] = {}
    for name in names:
        item = handle[name]
        if is_group(item):
            members[name] = _read_tree(item, is_group, data=data)
        else:
            samples = _value(item[()]) if data else None
            members[name] = Dataset(samples, {key: _value(value) for key, value in item.attrs.items()})
    return Group(members, {key: _value(value) for key, value in handle.attrs.items()})


def _value(value: Any) -> Any:
    """A read value as returned: bytes and NumPy strings as ``str``, everything else as a NumPy array."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    # pyfive hands back views of a memory map of the file. A copy keeps no reference to it, so the file
    # can be replaced or removed once read -- which Windows refuses while a map is open.
    return array if array.flags.owndata else array.copy()


def update_attrs(path: str | Path, attrs: Mapping[str, Mapping[str, Any]], *, backend: Backend = "auto") -> None:
    """Set attributes on members of an existing HDF5 file; the samples and every other attribute are kept.

    With h5py the file is changed in place. Without it the whole file is read, the attributes merged, and the
    result written beside the file and moved over it once complete, so a failed update leaves the file as it
    was; the rewritten file keeps the original's permissions, and a symlink is written through to its target.
    The built-in writer does not compress and stores a string dataset fixed-length, so a file h5py wrote comes
    back uncompressed when it is updated without h5py, holding the same values.

    Args:
        path: The file to change.
        attrs: Attributes to set, keyed by the path of the member that carries them: ``""`` for the file
            itself, ``"H1:STRAIN"`` for a top-level dataset, ``"group/dataset"`` below a group. An attribute
            already present is replaced; the others are kept.
        backend: ``"h5py"``, ``"builtin"``, or ``"auto"`` for h5py when it is installed and the built-in writer
            otherwise.

    Raises:
        FileNotFoundError: If there is no file at ``path``.
        KeyError: If a member path names nothing in the file; the file is not changed.
        ModuleNotFoundError: If ``backend="h5py"`` and h5py is not installed.
        TypeError: If the built-in writer cannot write a value the file holds, such as a half-precision
            float; the file is not changed.
        ValueError: If a member path has an empty, ``.`` or ``..`` component, or a string contains a NUL
            character.
    """
    targets = {member: _member_parts(member) for member in attrs}
    # Every value is checked and every member found before anything is changed, so a bad value or a
    # missing member changes nothing.
    converted = {
        member: {key: _attribute(key, value) for key, value in values.items()} for member, values in attrs.items()
    }
    h5py = _h5py() if backend in ("auto", "h5py") else None
    if backend == "h5py" and h5py is None:
        raise ModuleNotFoundError(
            "The h5py backend needs h5py, which is not installed. Install it with: pip install 'gwmock-data[hdf5]'",
            name="h5py",
        )
    if h5py is not None:
        with h5py.File(Path(path), "r+") as handle:
            found = {}
            for member, parts in targets.items():
                if parts and "/".join(parts) not in handle:
                    raise KeyError(f"The file has no member {member!r}.")
                found[member] = handle["/".join(parts)] if parts else handle
            for member, values in converted.items():
                for key, value in values.items():
                    found[member].attrs[key] = value
        return
    root = read(path, backend="builtin")
    found_nodes = {member: _find(root, member, parts) for member, parts in targets.items()}
    for member, values in converted.items():
        found_nodes[member].attrs.update(values)
    destination = Path(path)
    target = destination.resolve() if destination.is_symlink() else destination
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        write(root, partial, backend="builtin")
        shutil.copymode(target, partial)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def _attribute(key: str, value: Any) -> Any:
    """An attribute value as h5py stores it: a ``str`` as it is, anything else as a NumPy array.

    Checked here, before the file is touched, so that a value h5py would refuse half-way through an update
    changes nothing. An array of strings is stored as bytes (``dtype="S"``); a NumPy array of ``str``, an
    object array or a date is refused.

    Raises:
        TypeError: If the value is not a number, a string, or an array of numbers or bytes.
        ValueError: If a string contains a NUL character.
    """
    if isinstance(value, str):
        return _text(key, value)
    array = _array(value)
    if array.dtype.kind in "OUMm":
        raise TypeError(
            f"Cannot store an attribute of type {type(value).__name__} with dtype {array.dtype}; "
            "store an array of strings as bytes."
        )
    return array


def _member_parts(member: str) -> list[str]:
    """The components of a member path; ``""`` (the file itself) has none."""
    if member == "":
        return []
    parts = member.split("/")
    for part in parts:
        _check_name(part)
    return parts


def _find(root: Group, member: str, parts: list[str]) -> Group | Dataset:
    """The member of ``root`` that ``parts`` names; :class:`KeyError` if there is none."""
    node: Group | Dataset = root
    for part in parts:
        if not isinstance(node, Group) or part not in node.members:
            raise KeyError(f"The file has no member {member!r}.")
        node = node.members[part]
    return node
