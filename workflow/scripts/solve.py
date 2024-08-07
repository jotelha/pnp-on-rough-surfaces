input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

# debye_length = config["debye_length"]
potential_bias = config["potential_bias"]
reference_concentrations = config["reference_concentrations"]
number_charges = config["number_charges"]
number_of_species = config["number_of_species"]

mesh_msh = input.mesh_msh

solution_checkpoint_bp = output.solution_checkpoint_bp

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import numpy as np
import adios4dolfinx
import dolfinx.mesh
from matscipy.electrochemistry.poisson_nernst_planck_solver_2d_fenicsx import PoissonNernstPlanckSystemFEniCSx2d

# define desired system
pnp_2d = PoissonNernstPlanckSystemFEniCSx2d(
    c=reference_concentrations, z=number_charges,
    mesh_file=mesh_msh, scale_mesh=False)

for i in range(number_of_species):
    pnp_2d.apply_concentration_dirichlet_bc(i, pnp_2d.c_scaled[i], 2)

delta_u_scaled = potential_bias / pnp_2d.u_unit

pnp_2d.apply_potential_dirichlet_bc(delta_u_scaled, 4)
pnp_2d.apply_potential_dirichlet_bc(0.0, 2)

pnp_2d.solve()

adios4dolfinx.write_mesh(solution_checkpoint_bp, pnp_2d.mesh)
adios4dolfinx.write_function(filename=solution_checkpoint_bp, u=pnp_2d.w, name="solution")

# write mesh tags
meshtags = {}
for i in range(pnp_2d.mesh.topology.dim + 1):
    e_map = pnp_2d.mesh.topology.index_map(i)
    # Compute midpoints of entities
    entities = np.arange(e_map.size_local, dtype=np.int32)

    # Associate each local index with its global index
    values = np.arange(e_map.size_local, dtype=np.int32) + e_map.local_range[0]

    meshtags[i] = dolfinx.mesh.meshtags(pnp_2d.mesh, i, entities, values)

for i, tag in meshtags.items():
    adios4dolfinx.write_meshtags(filename=solution_checkpoint_bp,
                                 mesh=pnp_2d.mesh, meshtags=tag, meshtag_name=f"meshtags_{i}")

adios4dolfinx.write_meshtags(filename=solution_checkpoint_bp,
                             mesh=pnp_2d.mesh, meshtags=pnp_2d.facet_markers, meshtag_name="facet_markers")
adios4dolfinx.write_meshtags(filename=solution_checkpoint_bp,
                             mesh=pnp_2d.mesh, meshtags=pnp_2d.cell_markers, meshtag_name="cell_markers")
