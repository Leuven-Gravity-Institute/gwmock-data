"""Reading only the attributes of a file, and changing them in place."""

from __future__ import annotations

import stat
import sys

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
_BACKENDS = ["builtin", pytest.param("h5py", marks=_needs_h5py)]


def _strain_file(path):
    """A file shaped like a strain artifact: a top-level dataset, a nested one, attributes on all three."""
    hdf5.write(
        Group(
            {
                "H1:STRAIN": Dataset(np.arange(8.0), {"x0": 100.0, "unit": "strain"}),
                "metadata": Group({"record": Dataset("{}", {"version": 1})}, {"kind": "provenance"}),
            },
            {"schema": "gwmock-strain"},
        ),
        path,
        backend="builtin",
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_attributes_only_reads_no_samples(tmp_path, backend):
    """Every dataset comes back with ``data=None`` and its attributes, at any depth."""
    path = tmp_path / "strain.h5"
    _strain_file(path)

    root = hdf5.read(path, backend=backend, data=False)

    assert root.attrs["schema"] == "gwmock-strain"
    assert root.members["H1:STRAIN"].data is None
    assert float(root.members["H1:STRAIN"].attrs["x0"]) == 100.0
    record = root.members["metadata"].members["record"]
    assert record.data is None
    assert int(record.attrs["version"]) == 1


def test_attributes_only_never_touches_the_samples(tmp_path, monkeypatch):
    """The point of ``data=False``: no dataset is indexed."""
    pyfive = pytest.importorskip("pyfive")
    path = tmp_path / "strain.h5"
    _strain_file(path)

    def refuse(self, key):
        raise AssertionError(f"read the samples of {self.name}")

    monkeypatch.setattr(pyfive.Dataset, "__getitem__", refuse)

    hdf5.read(path, backend="builtin", data=False)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_attributes_are_set_and_everything_else_kept(tmp_path, backend):
    """New attributes appear, named ones are replaced, and samples and other attributes stay."""
    path = tmp_path / "strain.h5"
    _strain_file(path)

    hdf5.update_attrs(
        path,
        {"": {"run": '{"seed": 1}'}, "H1:STRAIN": {"dx": 0.25, "x0": 101.0}, "metadata/record": {"version": 2}},
        backend=backend,
    )

    root = hdf5.read(path, backend="builtin")
    assert root.attrs == {"schema": "gwmock-strain", "run": '{"seed": 1}'}
    strain = root.members["H1:STRAIN"]
    np.testing.assert_array_equal(strain.data, np.arange(8.0))
    assert {key: str(value) for key, value in strain.attrs.items()} == {"x0": "101.0", "unit": "strain", "dx": "0.25"}
    assert int(root.members["metadata"].members["record"].attrs["version"]) == 2
    assert root.members["metadata"].attrs == {"kind": "provenance"}


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("member", ["L1:STRAIN", "metadata/absent", "H1:STRAIN/deeper"])
def test_a_missing_member_changes_nothing(tmp_path, backend, member):
    """Every member is found before any attribute is set."""
    path = tmp_path / "strain.h5"
    _strain_file(path)
    before = path.read_bytes()

    with pytest.raises(KeyError, match=member):
        hdf5.update_attrs(path, {"": {"run": "x"}, member: {"a": 1}}, backend=backend)

    assert path.read_bytes() == before


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("member", ["/H1:STRAIN", "metadata//record", "./H1:STRAIN", "metadata/.."])
def test_a_malformed_member_path_is_refused(tmp_path, backend, member):
    """Paths are relative and have no empty, ``.`` or ``..`` components."""
    path = tmp_path / "strain.h5"
    _strain_file(path)

    with pytest.raises(ValueError, match="cannot name an HDF5 group member"):
        hdf5.update_attrs(path, {member: {"a": 1}}, backend=backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_a_file_that_does_not_exist_is_not_created(tmp_path, backend):
    """Updating is not writing: a missing file is an error, not a new empty file."""
    path = tmp_path / "absent.h5"

    with pytest.raises(FileNotFoundError):
        hdf5.update_attrs(path, {"": {"a": 1}}, backend=backend)

    assert not path.exists()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    "bad",
    [object(), np.array(["a", "b"]), np.array(["2026-10-18"], dtype="datetime64[D]")],
    ids=["object", "str array", "date"],
)
def test_a_value_hdf5_cannot_store_changes_nothing(tmp_path, backend, bad):
    """Values are checked before the file is touched, so a bad one leaves every attribute as it was."""
    path = tmp_path / "strain.h5"
    _strain_file(path)
    before = path.read_bytes()

    with pytest.raises(TypeError, match="Cannot store"):
        hdf5.update_attrs(path, {"": {"run": "x"}, "H1:STRAIN": {"bad": bad}}, backend=backend)

    assert path.read_bytes() == before


@pytest.mark.parametrize("backend", _BACKENDS)
def test_an_array_of_strings_is_stored_as_bytes(tmp_path, backend):
    """The supported spelling of a string array."""
    path = tmp_path / "strain.h5"
    _strain_file(path)

    hdf5.update_attrs(path, {"": {"detectors": np.array([b"H1", b"L1"])}}, backend=backend)

    assert hdf5.read(path, data=False).attrs["detectors"].tolist() == [b"H1", b"L1"]


@_needs_h5py
def test_a_file_the_built_in_writer_cannot_rewrite_is_left_alone(tmp_path):
    """Without h5py the file is rewritten; one holding a type the writer refuses is not touched."""
    path = tmp_path / "half.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("half", data=np.ones(3, dtype=np.float16))
    before = path.read_bytes()

    with pytest.raises(TypeError, match="float16"):
        hdf5.update_attrs(path, {"": {"a": 1}}, backend="builtin")

    assert path.read_bytes() == before
    assert [item.name for item in tmp_path.iterdir()] == ["half.h5"]


@_needs_h5py
def test_a_compressed_file_updated_without_h5py_keeps_its_values(tmp_path):
    """The rewrite drops the compression but not the samples."""
    path = tmp_path / "gzip.h5"
    hdf5.write(Group({"x": Dataset(np.linspace(0.0, 1.0, 1000))}), path, backend="h5py", compression="gzip")

    hdf5.update_attrs(path, {"x": {"unit": "strain"}}, backend="builtin")

    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["x"][()], np.linspace(0.0, 1.0, 1000))
        # A variable-length string, as h5py itself writes one, so h5py hands it back as ``str``.
        assert handle["x"].attrs["unit"] == "strain"
    assert hdf5.read(path, data=False).members["x"].attrs["unit"] == "strain"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits and symlinks")
def test_the_rewrite_keeps_permissions_and_writes_through_a_symlink(tmp_path):
    """Without h5py the file is replaced; it keeps its mode, and a link stays a link."""
    actual = tmp_path / "store" / "strain.h5"
    actual.parent.mkdir()
    _strain_file(actual)
    actual.chmod(0o600)
    alias = tmp_path / "strain.h5"
    alias.symlink_to(actual)

    hdf5.update_attrs(alias, {"": {"run": "x"}}, backend="builtin")

    assert alias.is_symlink()
    assert stat.S_IMODE(actual.stat().st_mode) == 0o600
    assert hdf5.read(actual, data=False).attrs["run"] == "x"


@pytest.mark.parametrize("backend", _BACKENDS)
def test_what_is_read_holds_no_reference_to_the_file(tmp_path, backend):
    """Every array read owns its memory, so nothing keeps the file mapped once ``read`` returns."""
    path = tmp_path / "strain.h5"
    _strain_file(path)

    root = hdf5.read(path, backend=backend)

    arrays = [root.members["H1:STRAIN"].data, root.members["H1:STRAIN"].attrs["x0"]]
    arrays += [root.members["metadata"].members["record"].attrs["version"]]
    assert all(array.flags.owndata for array in arrays)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_a_string_holding_a_nul_changes_nothing(tmp_path, backend):
    """Refused before the file is touched, on either backend."""
    path = tmp_path / "strain.h5"
    _strain_file(path)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="contains a NUL character"):
        hdf5.update_attrs(path, {"": {"run": "a\0b"}}, backend=backend)

    assert path.read_bytes() == before
