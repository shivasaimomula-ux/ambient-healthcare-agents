#!/usr/bin/env bash
#
# Cloud Agent install script for the Ambient Healthcare Agents developer example.
#
# This meta-repository is a collection of Jupyter notebooks (ambient-provider.ipynb,
# ambient-patient.ipynb) plus two git submodules that hold the full applications.
# The self-hosted applications require multi-GPU hosts, hundreds of GB of storage,
# and NGC-gated NIM containers, so they are not started here. This script prepares
# the development experience that IS achievable on a CPU-only agent: fetching the
# submodule source and installing the Jupyter toolchain used to work with and render
# the example notebooks.
#
# The script is idempotent: it is safe to run repeatedly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Initializing application submodules (ambient-provider, ambient-patient)"
git submodule update --init --recursive

echo "==> Installing the Jupyter notebook toolchain"
python3 -m pip install --user --upgrade pip
python3 -m pip install --user \
  ipykernel \
  nbclient \
  nbformat \
  jupyter \
  nbconvert \
  jupyterlab

echo "==> Registering the Python 3 Jupyter kernel used by the notebooks"
python3 -m ipykernel install --user --name python3 --display-name "Python 3"

echo "==> Install complete"
