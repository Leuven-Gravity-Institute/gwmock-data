"""Write HDF5 files in the classic layout, with NumPy alone.

The classic layout is what the HDF5 library writes by default and what every reader understands, pure-Python
ones included: a version 0 superblock, version 1 object headers, and groups stored as a symbol table (a version
1 B-tree, one symbol-table node, and a local heap of member names). Data is stored contiguously, little-endian
and uncompressed. A string attribute is stored as h5py stores one, a variable-length UTF-8 string kept in a
global heap collection, so that its length is not bounded by the 64 KiB an object header message can hold.

The structures used and their byte layouts are those of the HDF5 File Format Specification, version 2.0
(sections II.A "Superblock", III.A "B-trees", III.B "Group Symbol Table Nodes", III.C "Symbol Table Entry",
III.D "Local Heaps", III.E "Global Heap", IV.A "Data Object Headers"). Each group's members fit in one symbol-table node: the file's
group leaf node K, a superblock field, is chosen large enough for the largest group, so every B-tree is a single
leaf node.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gwmock_data.hdf5 import Dataset, Group

_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_UNDEFINED = 0xFFFFFFFFFFFFFFFF
#: Local heap free-list offset meaning "no free block": libhdf5's ``H5HL_FREE_NULL``, which it checks for.
_NO_FREE_BLOCK = 1
#: Size of a symbol table entry with 8-byte offsets and lengths.
_ENTRY_SIZE = 40
#: Group internal node K. Every group's B-tree is one leaf, so only the space it reserves matters.
_INTERNAL_K = 16

#: Bytes in a double-precision float.
_FLOAT64_SIZE = 8
#: Largest object header message: its size is a 16-bit field.
_MAX_MESSAGE_SIZE = 0xFFFF
#: Smallest global heap collection the HDF5 library accepts.
_MIN_COLLECTION_SIZE = 4096

# Object header message types.
_DATASPACE, _DATATYPE, _FILL_VALUE, _LAYOUT, _ATTRIBUTE, _SYMBOL_TABLE = 0x01, 0x03, 0x05, 0x08, 0x0C, 0x11


def _pad8(data: bytes) -> bytes:
    """Pad ``data`` with zeros to a multiple of eight bytes."""
    return data + b"\0" * (-len(data) % 8)


def datatype_message(dtype: np.dtype) -> bytes:
    """Encode the datatype message for ``dtype`` (specification IV.A.2.d), version 1.

    Supported: little-endian booleans, signed and unsigned integers, 32- and 64-bit floats, fixed-length byte
    strings, and compound types of those.

    Raises:
        TypeError: If ``dtype`` cannot be written.
    """
    dtype = np.dtype(dtype)
    if dtype.fields is not None:
        return _compound_message(dtype)
    if dtype.kind == "b":
        dtype = np.dtype(np.uint8)
    if dtype.kind in "iu":
        bits = 0x08 if dtype.kind == "i" else 0x00  # bit 3: signed
        header = struct.pack("<B3BI", 0x10 | 0, bits, 0, 0, dtype.itemsize)
        return header + struct.pack("<HH", 0, 8 * dtype.itemsize)
    if dtype.kind == "f" and dtype.itemsize in (4, 8):
        if dtype.itemsize == _FLOAT64_SIZE:
            sign, exponent_location, exponent_size, mantissa_size, bias = 63, 52, 11, 52, 1023
        else:
            sign, exponent_location, exponent_size, mantissa_size, bias = 31, 23, 8, 23, 127
        # Byte 0: little-endian, no padding, mantissa normalised with an implied leading one (bits 4-5 = 2).
        header = struct.pack("<B3BI", 0x10 | 1, 0x20, sign, 0, dtype.itemsize)
        properties = struct.pack(
            "<HHBBBBI", 0, 8 * dtype.itemsize, exponent_location, exponent_size, 0, mantissa_size, bias
        )
        return header + properties
    if dtype.kind == "S":
        # Null-padded, UTF-8: what h5py writes for a NumPy bytes array.
        return struct.pack("<B3BI", 0x10 | 3, 0x01 | (1 << 4), 0, 0, dtype.itemsize)
    raise TypeError(f"Cannot write HDF5 data of dtype {dtype}.")


def _compound_message(dtype: np.dtype) -> bytes:
    """Encode a compound datatype, version 1 (specification IV.A.2.d, class 6)."""
    names = dtype.names or ()
    header = struct.pack("<BHBI", 0x10 | 6, len(names), 0, dtype.itemsize)
    members = b""
    for name in names:
        member_dtype, offset = dtype.fields[name][:2]
        if member_dtype.fields is not None or member_dtype.shape:
            raise TypeError(f"Compound member {name!r} must be a scalar of a plain dtype.")
        members += _pad8(name.encode("utf-8") + b"\0")
        # Offset, dimensionality, reserved, dimension permutation, reserved, four dimension sizes.
        members += struct.pack("<IB3xI4x4I", offset, 0, 0, 0, 0, 0, 0)
        members += datatype_message(member_dtype)
    return header + members


def dataspace_message(shape: tuple[int, ...]) -> bytes:
    """Encode the dataspace of ``shape``, version 2 (specification IV.A.2.b).

    Version 2 says outright whether a dataspace is scalar (``shape == ()``) or simple; with version 1, a rank-0
    dataspace is read by some readers as one element of shape ``(1,)``.
    """
    kind = 0 if not shape else 1
    return struct.pack("<BBBB", 2, len(shape), 0, kind) + b"".join(struct.pack("<Q", size) for size in shape)


def _variable_length_string_message() -> bytes:
    """Encode the datatype of a variable-length UTF-8 string, version 1 (specification IV.A.2.d, class 9).

    Type 1 (string), null-terminated, UTF-8; elements are 16 bytes (a length and a global heap ID), and the base
    type is a one-byte unsigned integer, as the HDF5 library writes it.
    """
    base = struct.pack("<B3BIHH", 0x10 | 0, 0, 0, 0, 1, 0, 8)
    return struct.pack("<B3BI", 0x10 | 9, 0x01, 0x01, 0, 16) + base


def attribute_message(name: str, value: np.ndarray | str, heap: dict[str, tuple[int, int]] | None = None) -> bytes:
    """Encode an attribute message, version 1 (specification IV.A.2.m).

    A ``str`` is written as a scalar variable-length string whose bytes are in the global heap collection at the
    address and index ``heap`` gives for it.
    """
    encoded_name = name.encode("utf-8") + b"\0"
    if isinstance(value, str):
        address, index = (heap or {})[value]
        datatype = _variable_length_string_message()
        dataspace = dataspace_message(())
        data = struct.pack("<IQI", len(value.encode("utf-8")), address, index)
    else:
        datatype = datatype_message(value.dtype)
        dataspace = dataspace_message(value.shape)
        data = _little_endian(value).tobytes()
    header = struct.pack("<BBHHH", 1, 0, len(encoded_name), len(datatype), len(dataspace))
    message = header + _pad8(encoded_name) + _pad8(datatype) + _pad8(dataspace) + data
    if len(_pad8(message)) > _MAX_MESSAGE_SIZE:
        raise ValueError(
            f"Attribute {name!r} is {len(data)} bytes; an attribute of the built-in writer must fit in "
            f"{_MAX_MESSAGE_SIZE} bytes (strings excepted). Store it as a dataset, or install h5py."
        )
    return message


def global_heap_collection(strings: list[str]) -> tuple[bytes, dict[str, int]]:
    """Encode a global heap collection holding ``strings`` (specification III.E); return it and their indices.

    Objects are numbered from 1 in the order given. Their reference count is 0, as the HDF5 library writes it for
    variable-length data, and the space left is one free-space object (index 0), whose size counts its own header.
    """
    indices = {text: index for index, text in enumerate(strings, start=1)}
    objects = b"".join(
        struct.pack("<HHIQ", index, 0, 0, len(text.encode("utf-8"))) + _pad8(text.encode("utf-8"))
        for text, index in indices.items()
    )
    size = max(_MIN_COLLECTION_SIZE, 16 + len(objects) + 16)
    free = size - 16 - len(objects)
    collection = b"GCOL" + struct.pack("<B3xQ", 1, size) + objects + struct.pack("<HHIQ", 0, 0, 0, free)
    return collection + b"\0" * (size - len(collection)), indices


def _little_endian_dtype(dtype: np.dtype) -> np.dtype:
    """``dtype`` with every numeric part explicitly little-endian; a compound keeps its offsets and size.

    Native order (``"="``) is converted too: on a big-endian host it is big-endian. Single-byte types and byte
    strings have no order and are unchanged.
    """
    if dtype.fields is None:
        return dtype.newbyteorder("<")
    return np.dtype(
        {
            "names": list(dtype.names or ()),
            "formats": [_little_endian_dtype(dtype.fields[name][0]) for name in dtype.names or ()],
            "offsets": [dtype.fields[name][1] for name in dtype.names or ()],
            "itemsize": dtype.itemsize,
        }
    )


def _little_endian(value: np.ndarray) -> np.ndarray:
    """Return ``value`` contiguous and little-endian, booleans as one-byte integers.

    The datatype message always declares little-endian, so the bytes must be little-endian too, compound fields
    included.
    """
    # Not `np.ascontiguousarray`, which turns a 0-d array (a scalar) into a 1-d one.
    value = np.asarray(value)
    if not value.flags.c_contiguous:
        value = value.copy(order="C")
    if value.dtype.kind == "b":
        return value.astype(np.uint8)
    little = _little_endian_dtype(value.dtype)
    return value if little == value.dtype else value.astype(little)


def object_header(messages: list[tuple[int, bytes]]) -> bytes:
    """Encode a version 1 object header holding ``messages`` (specification IV.A.1.a)."""
    body = b""
    for message_type, data in messages:
        padded = _pad8(data)
        body += struct.pack("<HHB3x", message_type, len(padded), 0) + padded
    # Version, reserved, message count, reference count, header size; then padding to eight-byte alignment.
    return struct.pack("<BBHII4x", 1, 0, len(messages), 1, len(body)) + body


class _File:
    """Bytes of the file being built, with the offsets of what has been placed."""

    def __init__(self, leaf_k: int) -> None:
        self.leaf_k = leaf_k
        self.buffer = bytearray(96)  # The superblock and the root group's entry, filled in last.
        #: Where each string attribute's bytes are: the global heap collection's address and the object index.
        self.heap: dict[str, tuple[int, int]] = {}

    def strings(self, strings: list[str]) -> None:
        """Place one global heap collection holding ``strings``, for the attributes that refer to them."""
        if not strings:
            return
        collection, indices = global_heap_collection(strings)
        address = self.place(collection)
        self.heap = {text: (address, index) for text, index in indices.items()}

    def place(self, data: bytes) -> int:
        """Append ``data`` at an eight-byte boundary and return its address."""
        self.buffer += b"\0" * (-len(self.buffer) % 8)
        address = len(self.buffer)
        self.buffer += data
        return address

    def dataset(self, dataset: Dataset) -> int:
        """Place a dataset's samples and object header; return the object header's address."""
        data = _little_endian(np.asarray(dataset.data))
        raw = data.tobytes()
        data_address = self.place(raw) if raw else _UNDEFINED
        messages = [
            (_DATASPACE, dataspace_message(data.shape)),
            (_DATATYPE, datatype_message(data.dtype)),
            # Fill value, version 2: allocated early, never written, no value defined.
            (_FILL_VALUE, struct.pack("<BBBB", 2, 1, 2, 0)),
            # Layout, version 3, contiguous: address and size of the samples.
            (_LAYOUT, struct.pack("<BBQQ", 3, 1, data_address, len(raw))),
        ]
        messages += [
            (_ATTRIBUTE, attribute_message(key, value, self.heap)) for key, value in _attributes(dataset.attrs)
        ]
        return self.place(object_header(messages))

    def group(self, group: Group) -> tuple[int, int, int]:
        """Place a group and everything in it; return its object header, B-tree and local heap addresses."""
        names = sorted(group.members, key=lambda name: name.encode("utf-8"))
        children = []
        for name in names:
            member = group.members[name]
            if hasattr(member, "members"):
                header, btree, heap = self.group(member)
                children.append((name, header, (btree, heap)))
            else:
                children.append((name, self.dataset(member), None))

        # Local heap: the empty name at offset 0, then every member name, each null-terminated and padded.
        heap_data = b"\0" * 8
        name_offsets = []
        for name, _, _ in children:
            name_offsets.append(len(heap_data))
            heap_data += _pad8(name.encode("utf-8") + b"\0")
        heap_data_address = self.place(heap_data)
        heap_address = self.place(
            b"HEAP" + struct.pack("<B3xQQQ", 0, len(heap_data), _NO_FREE_BLOCK, heap_data_address)
        )

        btree = b"TREE" + struct.pack("<BBHQQ", 0, 0, 1 if children else 0, _UNDEFINED, _UNDEFINED)
        if children:
            # One symbol-table node with room for 2K entries, and the B-tree leaf pointing at it; keys are heap
            # offsets, the first of the empty name and the last of the largest name in the node.
            entries = b""
            for (_, header, scratch), name_offset in zip(children, name_offsets, strict=True):
                entries += _symbol_table_entry(name_offset, header, scratch)
            node = b"SNOD" + struct.pack("<BBH", 1, 0, len(children)) + entries
            node += b"\0" * (8 + 2 * self.leaf_k * _ENTRY_SIZE - len(node))
            btree += struct.pack("<QQQ", 0, self.place(node), name_offsets[-1])
        btree += b"\0" * (24 + 2 * _INTERNAL_K * 8 + (2 * _INTERNAL_K + 1) * 8 - len(btree))
        btree_address = self.place(btree)

        messages = [(_SYMBOL_TABLE, struct.pack("<QQ", btree_address, heap_address))]
        messages += [(_ATTRIBUTE, attribute_message(key, value, self.heap)) for key, value in _attributes(group.attrs)]
        return self.place(object_header(messages)), btree_address, heap_address


def _symbol_table_entry(name_offset: int, header: int, scratch: tuple[int, int] | None) -> bytes:
    """Encode a symbol table entry (specification III.C); a group's caches its B-tree and heap addresses."""
    if scratch is None:
        return struct.pack("<QQII16x", name_offset, header, 0, 0)
    return struct.pack("<QQIIQQ", name_offset, header, 1, 0, *scratch)


def _attributes(attrs: dict[str, object]) -> list[tuple[str, np.ndarray | str]]:
    """Attributes as arrays, a ``str`` as it is; sorted by name for a reproducible file."""
    return [(key, value if isinstance(value, str) else np.asarray(value)) for key, value in sorted(attrs.items())]


def _string_attributes(group: Group) -> set[str]:
    """Every distinct ``str`` attribute value in the tree, on groups and datasets alike."""
    found = {value for value in group.attrs.values() if isinstance(value, str)}
    for member in group.members.values():
        if hasattr(member, "members"):
            found |= _string_attributes(member)
        else:
            found |= {value for value in member.attrs.values() if isinstance(value, str)}
    return found


def _largest_group(group: Group) -> int:
    """The most members any group in the tree has."""
    sizes = [len(group.members)]
    sizes += [_largest_group(member) for member in group.members.values() if hasattr(member, "members")]
    return max(sizes)


def encode(root: Group) -> bytes:
    """Return the bytes of an HDF5 file holding ``root``."""
    leaf_k = max(4, -(-_largest_group(root) // 2))
    file = _File(leaf_k)
    # Sorted so the file is reproducible; one heap object per distinct string, however many attributes hold it.
    file.strings(sorted(_string_attributes(root)))
    header, btree, heap = file.group(root)
    end = len(file.buffer)
    superblock = _SIGNATURE + struct.pack("<8B", 0, 0, 0, 0, 0, 8, 8, 0)
    superblock += struct.pack("<HHI", leaf_k, _INTERNAL_K, 0)
    superblock += struct.pack("<QQQQ", 0, _UNDEFINED, end, _UNDEFINED)
    superblock += _symbol_table_entry(0, header, (btree, heap))
    file.buffer[: len(superblock)] = superblock
    return bytes(file.buffer)
