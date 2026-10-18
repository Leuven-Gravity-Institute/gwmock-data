# gwmock-data

[![Python CI](https://github.com/Leuven-Gravity-Institute/gwmock-data/actions/workflows/ci.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/gwmock-data/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Leuven-Gravity-Institute/gwmock-data/main.svg)](https://results.pre-commit.ci/latest/github/Leuven-Gravity-Institute/gwmock-data/main)
[![Documentation Status](https://github.com/Leuven-Gravity-Institute/gwmock-data/actions/workflows/documentation.yml/badge.svg)](https://leuven-gravity-institute.github.io/gwmock-data/)
[![codecov](https://codecov.io/gh/Leuven-Gravity-Institute/gwmock-data/graph/badge.svg)](https://codecov.io/gh/Leuven-Gravity-Institute/gwmock-data)
[![PyPI Version](https://img.shields.io/pypi/v/gwmock-data)](https://pypi.org/project/gwmock-data/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwmock-data)](https://pypi.org/project/gwmock-data/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

Shared data structures for the gwmock packages: the types they pass between each
other, so every package reads and writes them the same way.

## Installation

```bash
pip install gwmock-data
```

## Development

```bash
uv sync --group dev
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).
