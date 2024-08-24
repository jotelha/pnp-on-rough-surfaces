input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

reference_concentrations = config["reference_concentrations"]
number_charges = config["number_charges"]
number_of_species = config["number_of_species"]
temperature = config["temperature"]
relative_permittivity = config["relative_permittivity"]

checkpoint_bp = input.solution_checkpoint_bp

json_file = output.json_file

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

from mpi4py import MPI

from utils import ionic_strength, lambda_D

faraday_constant = sc.value('Faraday constant')

I = ionic_strength(z=number_charges, c=reference_concentrations)

logger.info("Ionic strength: %g.", I)

debye_length = lambda_D(ionic_strength=I, temperature=temperature, relative_permittivity=relative_permittivity)

logger.info("Debye length: %g.", debye_length)

mesh = adios4dolfinx.read_mesh(checkpoint_bp,
                               comm=MPI.COMM_WORLD,
                               engine="BP4", ghost_mode=dolfinx.mesh.GhostMode.none)

single_element = basix.ufl.element('Lagrange', mesh.basix_cell(), 3)
elements = [single_element] * (1+number_of_species)
mixed_element = basix.ufl.mixed_element(elements)
function_space = dolfinx.fem.functionspace(mesh, mixed_element)

solution_function = dolfinx.fem.Function(function_space)

logger.info("Read function from file '%s'.", checkpoint_bp)
adios4dolfinx.read_function(filename=checkpoint_bp, u=solution_function, name="solution")

solution_functions = solution_function.split()

potential_function = solution_functions[0]
concentration_functions = solution_functions[1:]

# faraday_constant = sc.value('Faraday constant')

# dimensional charge density
#
# \begin{equation*}
#     \rho = F \cdot \sum_{i=1}^{M} z_i c_i = F I \cdot \sum_{i=1}^{M} z_i c_i^* = F I \rho^{*}
# \end{equation*}
#
# vs
#
# dimensionless charge density
#
# \begin{equation*}
#     \rho^{*} = \sum_{i=1}^{M} z_i c_i^*
# \end{equation*}

charge_density = 0
for i in range(number_of_species):
    charge_density += number_charges[i]*concentration_functions[i]

charge_expression = dolfinx.fem.form(charge_density * ufl.dx)
charge_local = dolfinx.fem.assemble_scalar(charge_expression)
charge = mesh.comm.allreduce(charge_local, op=MPI.SUM)

# dimensional vs dimensionless charge
#
# \begin{equation*}
#     Q = F I \lambda_D^3 \int_{A^{*}} \rho^*\, \mathrm{d}A^*
# \end{equation*}
#
charge_SI = faraday_constant*I*debye_length**3*charge

amount_of_substance = []
for i in range(number_of_species):
    amount_of_substance_expression = dolfinx.fem.form(concentration_functions[i]* ufl.dx)
    amount_of_substance_local = dolfinx.fem.assemble_scalar(amount_of_substance_expression)
    amount_of_substance.append(mesh.comm.allreduce(amount_of_substance_local, op=MPI.SUM))

data = {
            'profile': wildcards.profile,
            'charge': charge,
            'charge_SI': charge_SI,
            **{f'amount_of_substance_{i}': amount_of_substance[i] for i in range(number_of_species)}
       }

with open(json_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)
