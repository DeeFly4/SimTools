#!/usr/bin/env bash

# This script installs all necessary packages to build DUNE-FemPy.

#Make sure that script exits on failure, and that all commands are printed
set -e
set -x

SUDO=
if [ "$UID" != "0" ]; then
  echo "Using sudo for installation!"
  SUDO=sudo
fi

# Make sure we have updated URLs to packages etc.
$SUDO apt-get update -y

# system packages
#$SUDO apt-get install -y gfortran
# there seems to be a problem with OpenMPI, so we take MPICH
$SUDO apt-get update \
  && apt-get dist-upgrade --no-install-recommends --yes \
  && apt-get install --no-install-recommends --yes \
  autoconf \
  automake \
  cmake \
  bison \
  build-essential \
  ca-certificates \
  curl \
  flex \
  gcc \
  g++ \
  gfortran \
  git \
  git-lfs \
  gnuplot-nox \
  libadolc-dev \
  libalberta-dev \
  libarpack++2-dev \
  libboost-dev \
  libboost-program-options-dev \
  libboost-serialization-dev \
  libboost-system-dev \
  libeigen3-dev \
  libffi-dev \
  libgmp-dev \
  libgtest-dev \
  libisl-dev \
  libltdl-dev \
  libmpfr-dev \
  libmpfrc++-dev \
  libmuparser-dev \
  libopenblas-dev \
  libscotchmetis-dev \
  libscotchparmetis-dev \
  libsuitesparse-dev \
  libsundials-dev \
  libsuperlu-dev \
  libtinyxml2-dev \
  libtiff-dev \
  libtool \
  locales-all \
  mpi-default-bin \
  mpi-default-dev \
  ninja-build \
  openssh-client \
  pkg-config \
  petsc-dev \
  python3 \
  python3-dev \
  python3-matplotlib \
  python3-mpi4py \
  python3-numpy \
  python3-pip \
  python3-pytest \
  python3-setuptools \
  python3-scipy \
  python3-ufl \
  python3-venv \
  python3-wheel \
  vc-dev \
  && apt-get clean
