Source dtool dataset: 3d68f4a0-21fb-449e-9c10-8fb752c91c5e

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

## Plot with pyvista

Launch a display server before running snakemake, e.g. with

```bash
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
```