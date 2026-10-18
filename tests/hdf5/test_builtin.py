"""The built-in HDF5 writer and the pyfive reader, checked against h5py and GWpy.

What the built-in writer writes must read back identically in h5py (and GWpy, for time series), and what h5py
writes must read back identically through the built-in path. The h5py comparisons skip where h5py is not
installed (Python 3.15); the pyfive round trips run everywhere.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from gwmock_data import hdf5
from gwmock_data.hdf5 import Dataset, Group

try:
    import h5py
except ModuleNotFoundError as exc:  # Python 3.15: h5py publishes no wheel yet
    if exc.name != "h5py":
        raise
    h5py = None

_needs_h5py = pytest.mark.skipif(h5py is None, reason="h5py is not installed")

_RECORDS = np.zeros(3, dtype=[("mass_1", "<f8"), ("count", "<i4"), ("flag", "u1")])
_RECORDS["mass_1"] = [30.0, 20.5, 1e-30]
_RECORDS["count"] = [-1, 0, 2**31 - 1]
_RECORDS["flag"] = [0, 1, 255]

#: Datasets of every kind the writer supports, keyed by member name.
_DATA = {
    "f64": np.random.default_rng(0).normal(size=257),
    "f32": np.arange(12, dtype=np.float32).reshape(3, 4),
    "i8": np.array([-(2**63), 0, 2**63 - 1], dtype=np.int64),
    "i1": np.array([-128, 127], dtype=np.int8),
    "u2": np.array([0, 65535], dtype=np.uint16),
    "u8": np.array([2**64 - 1], dtype=np.uint64),
    "bool": np.array([True, False, True]),
    "bytes": np.array([b"H1", b"L1:STRAIN"]),
    "records": _RECORDS,
    "scalar": np.float64(2.5),
    "text": '{"seed": 7, "name": "\\u00e9t\\u00e9"}',
    "empty": np.zeros(0),
    "3d": np.arange(24.0).reshape(2, 3, 4),
}

_ATTRS = {"x0": 1_400_000_000.25, "dx": 1 / 4096, "n": np.int64(3), "unit": "strain", "names": np.array([1.0, 2.0])}


def _tree() -> Group:
    members = {name: Dataset(value, dict(_ATTRS)) for name, value in _DATA.items()}
    members["group"] = Group({"inner": Dataset(np.arange(3.0))}, {"note": "nested"})
    members["empty group"] = Group()
    return Group(members, {"schema": "1.0.0", "count": np.int32(2)})


def _expected(value):
    if isinstance(value, str):
        return value
    return np.asarray(value)


def _assert_same(read_value, written):
    expected = _expected(written)
    if isinstance(expected, str):
        assert read_value == expected
        return
    read_value = np.asarray(read_value)
    assert read_value.shape == expected.shape
    if expected.dtype.fields is not None:
        assert read_value.dtype.names == expected.dtype.names
        for name in expected.dtype.names:
            np.testing.assert_array_equal(read_value[name], expected[name])
    else:
        np.testing.assert_array_equal(read_value, expected)


def _assert_tree(read_root: Group):
    assert set(read_root.members) == set(_DATA) | {"group", "empty group"}
    for name, value in _DATA.items():
        member = read_root.members[name]
        _assert_same(member.data, value)
        for key, attr in _ATTRS.items():
            _assert_same(member.attrs[key], attr)
    inner = read_root.members["group"]
    np.testing.assert_array_equal(inner.members["inner"].data, np.arange(3.0))
    assert inner.attrs["note"] == "nested"
    assert read_root.members["empty group"].members == {}
    assert read_root.attrs["schema"] == "1.0.0"
    assert int(read_root.attrs["count"]) == 2


def test_the_built_in_writer_round_trips_through_pyfive(tmp_path):
    """Without h5py: the built-in writer, then pyfive, return every member and attribute unchanged."""
    hdf5.write(_tree(), tmp_path / "builtin.h5", backend="builtin")

    _assert_tree(hdf5.read(tmp_path / "builtin.h5", backend="builtin"))


@_needs_h5py
def test_h5py_reads_what_the_built_in_writer_writes(tmp_path):
    """h5py reads every value of a built-in file, with the dtypes and scalar shapes it was written with."""
    hdf5.write(_tree(), tmp_path / "builtin.h5", backend="builtin")

    _assert_tree(hdf5.read(tmp_path / "builtin.h5", backend="h5py"))
    with h5py.File(tmp_path / "builtin.h5", "r") as handle:
        # Native dtypes, not just values: integers keep their width and sign, floats their precision.
        for name in ("f32", "i8", "i1", "u2", "u8"):
            assert handle[name].dtype == _DATA[name].dtype
        assert handle["scalar"].shape == ()
        assert handle["text"].shape == ()


@_needs_h5py
def test_the_built_in_path_reads_what_h5py_writes_compressed(tmp_path):
    """Pyfive reads what h5py writes, gzip-compressed datasets included."""
    hdf5.write(_tree(), tmp_path / "h5py.h5", backend="h5py", compression="gzip")

    _assert_tree(hdf5.read(tmp_path / "h5py.h5", backend="builtin"))


@_needs_h5py
def test_gwpy_reads_a_time_series_the_built_in_writer_writes(tmp_path):
    """GWpy loads a series the built-in writer wrote, with its timing, unit and name."""
    timeseries = pytest.importorskip("gwpy.timeseries")
    samples = np.random.default_rng(1).normal(size=4096)
    attrs = {"x0": 1_400_000_000.0, "dx": 1 / 4096, "xunit": "s", "unit": "strain", "name": "H1:STRAIN"}
    hdf5.write(Group({"H1:STRAIN": Dataset(samples, attrs)}), tmp_path / "ts.h5", backend="builtin")

    series = timeseries.TimeSeriesDict.read(tmp_path / "ts.h5", ["H1:STRAIN"], format="hdf5")["H1:STRAIN"]

    np.testing.assert_array_equal(series.value, samples)
    assert series.t0.value == 1_400_000_000.0
    assert series.dt.value == 1 / 4096
    assert str(series.unit) == "strain"
    assert series.name == "H1:STRAIN"


@pytest.mark.parametrize("count", [0, 1, 8, 9, 300])
def test_groups_of_any_size_round_trip(tmp_path, count):
    """Group members live in one symbol-table node sized for the largest group; check across sizes."""
    names = [f"m{index:04d}" for index in range(count)]
    root = Group({name: Dataset(np.array([float(index)])) for index, name in enumerate(names)})
    hdf5.write(root, tmp_path / "many.h5", backend="builtin")

    back = hdf5.read(tmp_path / "many.h5", backend="builtin")

    assert sorted(back.members) == names
    for index, name in enumerate(names):
        assert float(back.members[name].data[0]) == float(index)
    if h5py is not None:
        assert sorted(hdf5.read(tmp_path / "many.h5", backend="h5py").members) == names


def test_names_are_stored_as_given_and_read_in_byte_order(tmp_path):
    """Member names, non-ASCII included, are kept; HDF5 lists them in byte order."""
    names = ["b", "a", "B", "H1:STRAIN", "été", "with space"]
    hdf5.write(Group({name: Dataset(np.zeros(1)) for name in names}), tmp_path / "names.h5", backend="builtin")

    back = hdf5.read(tmp_path / "names.h5", backend="builtin")

    assert list(back.members) == sorted(names, key=lambda name: name.encode("utf-8"))


@pytest.mark.parametrize("backend", ["builtin", pytest.param("h5py", marks=_needs_h5py)])
@pytest.mark.parametrize("name", ["", "a/b", ".", ".."])
def test_a_name_hdf5_would_read_as_a_path_is_refused(tmp_path, name, backend):
    """Refused whichever backend writes, so the same tree never means a different file."""
    with pytest.raises(ValueError, match="cannot name an HDF5 group member"):
        hdf5.write(Group({name: Dataset(np.zeros(1))}), tmp_path / "bad.h5", backend=backend)
    assert not (tmp_path / "bad.h5").exists()


@pytest.mark.parametrize("order", [">", "<", "="])
def test_compound_rows_in_any_byte_order_keep_their_values(tmp_path, order):
    """The datatype declares little-endian, so big-endian fields must be converted, offsets kept."""
    dtype = np.dtype(
        {"names": ["i", "f", "b"], "formats": [f"{order}i4", f"{order}f8", "u1"], "offsets": [0, 8, 16], "itemsize": 24}
    )
    rows = np.zeros(2, dtype=dtype)
    rows["i"] = [0x01020304, -5]
    rows["f"] = [1.5, -2.25e300]
    rows["b"] = [7, 255]
    group = Group({"rows": Dataset(rows, {"row": rows[0:1]})})
    hdf5.write(group, tmp_path / "rows.h5", backend="builtin")

    backends = ["builtin"] + (["h5py"] if h5py is not None else [])
    for backend in backends:
        back = hdf5.read(tmp_path / "rows.h5", backend=backend).members["rows"]
        for name in ("i", "f", "b"):
            np.testing.assert_array_equal(back.data[name], rows[name])
            np.testing.assert_array_equal(np.asarray(back.attrs["row"])[name], rows[0:1][name])


@pytest.mark.parametrize("value", [np.zeros(2, dtype=np.complex128), np.array(["text"]), np.array([object()])])
def test_an_unsupported_dtype_is_refused(tmp_path, value):
    """A dtype the writer cannot encode is refused rather than written wrongly."""
    with pytest.raises(TypeError, match="Cannot write HDF5 data of dtype"):
        hdf5.write(Group({"x": Dataset(value)}), tmp_path / "bad.h5", backend="builtin")


def test_the_h5py_backend_without_h5py_names_the_extra(monkeypatch, tmp_path):
    """Asking for h5py explicitly without it installed says how to install it."""
    real_import = importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "h5py" or name.startswith("h5py."):
            raise ModuleNotFoundError("No module named 'h5py'", name="h5py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)

    for call in (
        lambda: hdf5.write(Group(), tmp_path / "x.h5", backend="h5py"),
        lambda: hdf5.read(tmp_path / "x.h5", backend="h5py"),
    ):
        with pytest.raises(ModuleNotFoundError, match=r"pip install 'gwmock-data\[hdf5\]'"):
            call()


def test_a_written_file_is_byte_for_byte_reproducible(tmp_path):
    """The same tree writes the same bytes, so a file hash means the content."""
    hdf5.write(_tree(), tmp_path / "a.h5", backend="builtin")
    hdf5.write(_tree(), tmp_path / "b.h5", backend="builtin")

    assert (tmp_path / "a.h5").read_bytes() == (tmp_path / "b.h5").read_bytes()


@pytest.mark.parametrize("order", ["<", ">", "="])
@pytest.mark.parametrize("kind", ["i2", "u4", "i8", "f4", "f8"])
def test_every_numeric_dtype_is_made_explicitly_little_endian(order, kind):
    """Checked on the dtype, so it holds on any host: native order means big-endian on a big-endian one."""
    from gwmock_data.hdf5._writer import _little_endian_dtype

    dtype = np.dtype(order + kind)
    compound = np.dtype([("x", dtype), ("label", "S4"), ("flag", "u1")])

    assert _little_endian_dtype(dtype) == np.dtype("<" + kind)
    assert _little_endian_dtype(dtype).str.startswith("<")
    converted = _little_endian_dtype(compound)
    assert converted.fields["x"][0].str == "<" + kind
    assert converted.fields["label"][0] == np.dtype("S4")
    assert [converted.fields[name][1] for name in ("x", "label", "flag")] == [
        compound.fields[name][1] for name in ("x", "label", "flag")
    ]
    assert converted.itemsize == compound.itemsize


@pytest.mark.parametrize(
    "dtype",
    [np.dtype([("inner", [("a", "<f8")])]), np.dtype([("vector", "<f8", (3,))])],
    ids=["nested compound", "array member"],
)
def test_a_compound_member_that_is_not_a_plain_scalar_is_refused(tmp_path, dtype):
    """Only scalar members of plain dtypes are encoded; anything else is refused, not flattened."""
    with pytest.raises(TypeError, match="must be a scalar of a plain dtype"):
        hdf5.write(Group({"rows": Dataset(np.zeros(2, dtype=dtype))}), tmp_path / "bad.h5", backend="builtin")


def test_a_non_contiguous_array_is_written_in_order(tmp_path):
    """A strided view is copied into C order before its bytes are written."""
    base = np.arange(20.0).reshape(4, 5)
    view = base[:, ::2]
    assert not view.flags.c_contiguous
    hdf5.write(Group({"view": Dataset(view)}), tmp_path / "view.h5", backend="builtin")

    np.testing.assert_array_equal(hdf5.read(tmp_path / "view.h5", backend="builtin").members["view"].data, view)


@pytest.mark.parametrize("backend", ["builtin", pytest.param("h5py", marks=_needs_h5py)])
def test_reading_only_some_members(tmp_path, backend, monkeypatch):
    """Only the named top-level members are read, in the order asked; absent names are skipped."""
    hdf5.write(_tree(), tmp_path / "tree.h5", backend="builtin")

    back = hdf5.read(tmp_path / "tree.h5", backend=backend, members=["i8", "group", "missing", "i8"])

    assert list(back.members) == ["i8", "group"]
    _assert_same(back.members["i8"].data, _DATA["i8"])
    np.testing.assert_array_equal(back.members["group"].members["inner"].data, np.arange(3.0))
    assert back.attrs["schema"] == "1.0.0"


def test_members_not_asked_for_are_never_read(tmp_path, monkeypatch):
    """The point of selecting: an unrequested member's data is not touched."""
    pyfive = pytest.importorskip("pyfive")
    hdf5.write(_tree(), tmp_path / "tree.h5", backend="builtin")
    read_names = []
    original = pyfive.Dataset.__getitem__

    def recording(self, key):
        read_names.append(self.name)
        return original(self, key)

    monkeypatch.setattr(pyfive.Dataset, "__getitem__", recording)
    hdf5.read(tmp_path / "tree.h5", backend="builtin", members=["f64"])

    assert [name.strip("/") for name in read_names] == ["f64"]


@pytest.mark.parametrize("members", ["data", b"data"])
def test_a_single_string_of_members_is_refused(tmp_path, members):
    """A bare string would be read as one-character names and select nothing, silently."""
    hdf5.write(Group({"data": Dataset(np.zeros(2))}), tmp_path / "one.h5", backend="builtin")

    with pytest.raises(TypeError, match="collection of names"):
        hdf5.read(tmp_path / "one.h5", members=members)


@pytest.mark.parametrize("backend", ["builtin", pytest.param("h5py", marks=_needs_h5py)])
@pytest.mark.parametrize("name", ["group/inner", "", ".", ".."])
def test_a_member_path_is_refused(tmp_path, backend, name):
    """Only top-level members can be selected; a path into a group would come back flattened under its path."""
    hdf5.write(_tree(), tmp_path / "tree.h5", backend="builtin")

    with pytest.raises(ValueError, match="cannot name an HDF5 group member"):
        hdf5.read(tmp_path / "tree.h5", backend=backend, members=[name])


def test_members_may_be_a_one_shot_iterator(tmp_path):
    """The names are checked and then used; a generator must survive both."""
    hdf5.write(_tree(), tmp_path / "tree.h5", backend="builtin")

    back = hdf5.read(tmp_path / "tree.h5", backend="builtin", members=(name for name in ["i8"]))

    assert list(back.members) == ["i8"]
