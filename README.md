## Simulation Tools (FMMN05)

Code and project reports for the course Simulation Tools at LTH.

## Requirements

The code is Python used with conda environments. All projects use [Assimulo](https://jmodelica.org/assimulo/) and some other basic packages. You may have to install a specific **python** version for compatibility.

### Projects 1 and 2

```console
conda create --name simtools
conda activate simtools
conda install -c conda-forge matplotlib scipy numpy
conda install -c conda-forge assimulo
```

### Project 3

This project uses [DUNE](https://dune-project.org/) as well. This software will only install and run on **Linux** and **MacOS**. **Windows** users can use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to install **Ubuntu LTS**. A collection of necessary **Ubuntu** packages can be found in [bootstrap.sh](bootstrap.sh). The installation has to be done with root privileges, e.g.

```console
sudo bootstrap.sh
```

This takes care of the prerequisities to install DUNE. Now create a conda environment:

```console
conda create -n duneproject
conda activate duneproject
conda install -c anaconda ipython
conda install -c conda-forge mpich mpich-mpicc mpich-mpicxx
conda install -c conda-forge assimulo matplotlib
conda install -c conda-forge fenics-ufl pkg-config cmake
```

And then install DUNE with **pip**:

```console
pip install mpi4py
pip install --pre dune-fem
```