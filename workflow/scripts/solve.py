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

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import adios4dolfinx
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

adios4dolfinx.write_mesh(output.solution_checkpoint_bp, pnp_2d.mesh)
adios4dolfinx.write_function(filename=output.solution_checkpoint_bp, u=pnp_2d.w, name="solution")