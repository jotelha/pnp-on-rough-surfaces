input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

debye_length = config["debye_length"]
potential_bias = config["potential_bias"]
reference_concentrations = config["reference_concentrations"]
number_charges = config["number_charges"]
number_of_species = config["number_of_species"]

dimensional_solution_checkpoint_bp = input.dimensional_solution_checkpoint_bp

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import json

import basix
import dolfinx
import ufl
import adios4dolfinx

import numpy as np
import scipy.constants as sc

from dolfinx import default_scalar_type

from mpi4py import MPI

mesh = adios4dolfinx.read_mesh(dimensional_solution_checkpoint_bp,
                               comm=MPI.COMM_WORLD,
                               engine="BP4", ghost_mode=dolfinx.mesh.GhostMode.none)

single_element_CG1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
scalar_function_space_CG1 = dolfinx.fem.functionspace(mesh, single_element_CG1)
potential_function = dolfinx.fem.Function(scalar_function_space_CG1,
                                          dtype=dolfinx.default_scalar_type)
concentration_functions = [dolfinx.fem.Function(scalar_function_space_CG1,
                                          dtype=dolfinx.default_scalar_type) for _ in range(number_of_species)]

adios4dolfinx.read_function(
        filename=input.dimensional_solution_checkpoint_bp,
        u=potential_function, name="potential")

for i in range(number_of_species):
    adios4dolfinx.read_function(
        filename=input.dimensional_solution_checkpoint_bp,
        u=concentration_functions[i], name=f"concentration_{i}")

faraday_constant = sc.value('Faraday constant')

charge_density_SI = 0
for i in range(number_of_species):
    charge_density_SI += faraday_constant*number_charges[i]*concentration_functions[i]

charge_SI_expression = dolfinx.fem.form(charge_density_SI * ufl.dx)
charge_SI_local = dolfinx.fem.assemble_scalar(charge_SI_expression)
charge_SI = mesh.comm.allreduce(charge_SI_local, op=MPI.SUM)

data = {
            'profile': wildcards.profile,
            'charge_SI': charge_SI,
        }

with open(output.json_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)

# compute 1d reference solution
# pnp_1d = PoissonNernstPlanckSystemFEniCSx(c=c, z=z, delta_u=delta_u, L=L, N=1000)
# pnp_1d.use_standard_interface_bc()
#
# potential_ref_normalized, concentrations_ref_normalized, _ = pnp_1d.solve()
#
# x_ref_unitless = pnp_1d.grid_dimensionless
#
# for i, c_ref_normalized in enumerate(concentrations_ref_normalized):
#     N_excess_ref = np.trapz(c_ref_normalized-1., x_ref_unitless)
