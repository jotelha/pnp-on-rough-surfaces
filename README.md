# Poisson-Nernst-Planck equation on rough surfaces

## Input

Raw profiles are found within `in/profiles`.

Profile-specific configuration and assignment of unique labels happens within `workflow/config.yml`.

## Output

Content of `output directory`:

``
all.csv  # homogenized global properties of all systems
figures  # publication-quality figures of global properties
geometries  # 2d geometries generated from line scan profiles
global_properties_plots  # plots of global properties
meshes  # meshes generated from geometries
potential_0.005  # sweep across potential values
potential_0.01
...
potential_0.18
potential_0.2
profiles  # homogenized line scan profiles
```

Within each `potential_*` subfolder, the following folders contain polished figures:

* plot_solution_1d: illustration of 1d double layer
* plot_solution_2d_global: color maps of potential and concentrations on whole domain
* plot_solution_2d_local: color maps of potential and concentrations on small subdomain

* surface_excess_global_plots: surface excess on whole domain
* surface_excess_global_with_gpr_plots: surface excess with delocalized trends on whole domain
* surface_excess_local_plots: surface excess on small subdomain
* surface_excess_local_with_gpr_plots: surface excess with delocalized trends on small subdomain

* comparative_plots: plots comparing global scalar profile-specific properties

Other folders contain intermediate or final results.

* checkpoint: immediate, dimensionless solution of PNP system
* comparative_properties: global scalar profile-specific properties with respect to reference profile
* derived_properties: global scalar profile-specific properties derived from FEM solution
* geometries: domain geometries generated from profiles
* interpolation: dimensionless solution of PNP system interpolated down to 1st order Lagrange elements for visualization
meshes
* profile_properties: global scalar profile-specific properties derived from profiles alone
profiles
* surface_charge: results from surface charge compuaton
* surface_integrals: results from integration of variables in z direction
* volume_integrals: results from integration over whole domain

## Installation

To install development versions of git repositories within container, 
mark dirctory as safe, i.e. with

    git config --global --add safe.directory /tmp/data/matscipy

before running

    pip install .

For an editable install of matscipy, use

    pip install meson-python
    pip install --no-build-isolation -e .

## Run with snakemake

Use

```bash
snakemake --cores all --verbose
```

to run all analysis.

Create directed acyclic graphs of the workflow with

```bash
snakemake --dag | dot -Tsvg > dag_all.svg
snakemake --rulegraph | dot -Tsvg > rulegraph.svg
snakemake --filegraph | dot -Tsvg > filegraph.svg
```

## Snakemake interactive notebook editing

To be able to interactively edit a notebook with a command like

```bash
snakemake --cores 1 --edit-notebook out/figures/surplus_surface_excess_potential_bias.svg --notebook-listen 0.0.0.0:8889
```

launch the conainer embedding snakemake with additional ports open first, e.g.

```bash
docker run -v $(pwd):/tmp/data --init -ti -p 8888:8888 -p 8889:8889 imteksim/dolfinx-tutorial-extended
```

Generate a jupyter configuration with

```console
# jupyter notebook --generate-config
Writing default config to: /root/.jupyter/jupyter_notebook_config.py
```

and append the lines

```bash
echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py
# echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
```

to the config file.

## Plot with pyvista

Launch a display server before running snakemake, e.g. with

```bash
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
```

## Plot with matplotlib

Use `matplotlibrc` in repository with

```bash
export MATPLOTLIBRC="$(pwd)/matplotlibrc"
```

before running plotting scripts.

Install Arial with

```bash
apt update
apt install ttf-mscorefonts-installer
```
