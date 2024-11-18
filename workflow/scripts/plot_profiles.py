input = snakemake.input
output = snakemake.output
config = snakemake.config

mesh_msh = input.mesh_msh
json_file = input.json_file
png_file = output.png_file
svg_file = output.svg_file

relative_lateral_cutoff = config["relative_lateral_cutoff"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gmsh

x_axis_label = 'length x ($\lambda_D)$'
h_axis_label = 'profile height h ($\lambda_D)$'

import json

with open(json_file, 'r') as file:
    metadata = json.load(file)

rms_height_SI = metadata["rms_height_SI"]

# get boundary
gmsh.initialize()

# Open the mesh file
gmsh.open(mesh_msh)

# get nodes on rough edge
nodes, coords = gmsh.model.mesh.getNodesForPhysicalGroup(1, 4)

coords_3d = np.reshape(coords, (len(nodes),3))

sorted_indices = np.argsort(coords_3d[:,0])

coords_3d = coords_3d[sorted_indices,:]

df = pd.DataFrame({'x': coords_3d[:,0], 'y': coords_3d[:,1]})

xmin = df["x"].min()
xmax = df["x"].max()

lateral_span = (xmax - xmin)
discarded_dx = lateral_span * relative_lateral_cutoff

lower_boundary = xmin + discarded_dx
upper_boundary = xmax - discarded_dx

df = df[(df["x"] > lower_boundary) & (df["x"] < upper_boundary)]

fig, ax1 = plt.subplots(1,1, figsize=None)

color = 'dimgray'
ax1.set_xlabel(x_axis_label)
ax1.set_ylabel(h_axis_label, color=color)

# plot roughness profile
ax1.plot(
    df["x"], df["y"],
    color=color, linestyle="-", linewidth=1, label="roughness profile")

ax1.set_title(f"Rq = {rms_height_SI*1e9:.1f} nm")

fig.set_size_inches(7, 2.5)

#  7.0, 5.25
fig.tight_layout()
fig.savefig(svg_file)
fig.savefig(png_file)
