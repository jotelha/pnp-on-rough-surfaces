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

solution_checkpoint_bp = input.solution_checkpoint_bp

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import basix
import dolfinx
import adios4dolfinx

import numpy as np

from dolfinx import default_scalar_type
from dolfinx.fem import Function, FunctionSpace

from mpi4py import MPI

mesh = adios4dolfinx.read_mesh(MPI.COMM_WORLD, solution_checkpoint_bp, "BP4", dolfinx.mesh.GhostMode.none)

P = basix.ufl.element('Lagrange', mesh.basix_cell(), 3)
elements = [P] * 3
H = basix.ufl.mixed_element(elements)
W = dolfinx.fem.FunctionSpace(mesh, H)

w = dolfinx.fem.Function(W)

adios4dolfinx.read_function(w, solution_checkpoint_bp)

solution_functions = w.split()
potential_function = solution_functions[0]
concentration_functions = solution_functions[1:]

gdim = mesh.geometry.dim

H0 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
W0 = FunctionSpace(mesh, H0)

potential_function_normalized_interpolated = Function(W0, dtype=default_scalar_type)
potential_function_normalized_interpolated.interpolate(potential_function)

concentration_functions_normalized_interpolated = []

M = len(concentration_functions)
for i, concentration_function in enumerate(concentration_functions):
    concentration_function_normalized_interpolated = Function(W0, dtype=default_scalar_type)
    concentration_function_normalized_interpolated.interpolate(concentration_function)
    concentration_functions_normalized_interpolated.append(concentration_function_normalized_interpolated)

# compute 1d reference solution
pnp_1d = PoissonNernstPlanckSystemFEniCSx(c=c, z=z, delta_u=delta_u, L=L, N=1000)
pnp_1d.use_standard_interface_bc()

potential_ref_normalized, concentrations_ref_normalized, _ = pnp_1d.solve()

x_ref_unitless = pnp_1d.grid_dimensionless

for i, c_ref_normalized in enumerate(concentrations_ref_normalized):
    N_excess_ref = np.trapz(c_ref_normalized-1., x_ref_unitless)
