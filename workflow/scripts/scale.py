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
temperature = config["temperature"]
relative_permittivity = config["relative_permittivity"]

interpolated_solution_checkpoint_bp = input.interpolated_solution_checkpoint_bp
dimensional_solution_checkpoint_bp = output.dimensional_solution_checkpoint_bp
dimensional_potential_xdmf = output.dimensional_potential_xdmf
dimensional_concentration_xdmfs = output[0:number_of_species]

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import basix
import dolfinx
import adios4dolfinx

from dolfinx.io import XDMFFile
from mpi4py import MPI

import numpy as np
import scipy.constants as sc

vacuum_permittivity = sc.epsilon_0
gas_constant = sc.value('molar gas constant')
faraday_constant = sc.value('Faraday constant')

# default output settings
label_width = 40  # character width of quantity labels in log

f = faraday_constant / (gas_constant * temperature)  # for convenience

# print all quantities to log
logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
    'temperature T', temperature, lwidth=label_width))
logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
    'relative permittivity eps_R', relative_permittivity, lwidth=label_width))
logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
    'vacuum permittivity eps_0', vacuum_permittivity, lwidth=label_width))
logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
    'universal gas constant R', gas_constant, lwidth=label_width))
logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
    'Faraday constant F', faraday_constant, lwidth=label_width))
logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
    'f = F / (RT)', f, lwidth=label_width))


def ionic_strength(z, c):
    """Compute a system's ionic strength from charges and concentrations.

    Returns
    -------
    ionic_strength : float
        ionic strength ( 1/2 * sum(z_i^2*c_i) )
        [concentration unit, i.e. mol m^-3]
    """
    return 0.5*np.sum(np.square(z) * c)


def lambda_D(ionic_strength):
    """Compute the system's Debye length.

    Returns
    -------
    lambda_D : float
        Debye length, sqrt( epsR*eps*R*T/(2*F^2*I) ) [length unit, i.e. m]
    """
    return np.sqrt(
        relative_permittivity * vacuum_permittivity * gas_constant * temperature / (
                2.0 * faraday_constant ** 2 * ionic_strength))


c_unit = ionic_strength(z=number_charges, c=reference_concentrations)
l_unit = lambda_D(ionic_strength=c_unit)
u_unit = gas_constant * temperature / faraday_constant

logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
    'spatial unit [l]', l_unit, lwidth=label_width))
logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
    'concentration unit [c]', c_unit, lwidth=label_width))
logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
    'potential unit [u]', u_unit, lwidth=label_width))


logger.info("Read mesh from file '%s'.", interpolated_solution_checkpoint_bp)
mesh = adios4dolfinx.read_mesh(interpolated_solution_checkpoint_bp,
                               comm=MPI.COMM_WORLD,
                               engine="BP4",
                               ghost_mode=dolfinx.mesh.GhostMode.none)

single_element_CG1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
scalar_function_space_CG1 = dolfinx.fem.functionspace(mesh, single_element_CG1)
potential_function = dolfinx.fem.Function(scalar_function_space_CG1,
                                          dtype=dolfinx.default_scalar_type)
logger.info("Read functions from file '%s'.", interpolated_solution_checkpoint_bp)

adios4dolfinx.read_function(interpolated_solution_checkpoint_bp,
                            potential_function,
                            name="solution_function_0")
# potential_function *= u_unit
potential_function.vector.array[:] = u_unit * potential_function.vector.array

concentration_functions = []
for i in range(number_of_species):
    concentration_function = dolfinx.fem.Function(
        scalar_function_space_CG1, dtype=dolfinx.default_scalar_type)
    adios4dolfinx.read_function(interpolated_solution_checkpoint_bp,
                                concentration_function,
                                name=f"solution_function_{i+1}")
    # concentration_function *= c_unit
    concentration_function.vector.array[:] = c_unit * concentration_function.vector.array
    concentration_functions.append(concentration_function)

mesh.geometry.x[:] = mesh.geometry.x * l_unit

adios4dolfinx.write_mesh(dimensional_solution_checkpoint_bp, mesh,
                         engine="BP4")

logger.info("Write potential to file '%s'.", dimensional_solution_checkpoint_bp)
adios4dolfinx.write_function(filename=dimensional_solution_checkpoint_bp,
                             u=potential_function,
                             name=f"potential")

for i, concentration_function in enumerate(concentration_functions):
    logger.info("Write concentration %d to file '%s'.", i,
                dimensional_solution_checkpoint_bp)
    adios4dolfinx.write_function(filename=dimensional_solution_checkpoint_bp,
                                 u=concentration_function,
                                 name=f"concentration_{i}")

logger.info("Write potential to file '%s'.", dimensional_potential_xdmf)
with XDMFFile(mesh.comm, dimensional_potential_xdmf, "w") as file:
    file.write_mesh(mesh)
    file.write_function(potential_function)

for i, concentration_function in enumerate(concentration_functions):
    logger.info("Write concentration %d to file '%s'.", i,
                dimensional_concentration_xdmfs[i])
    with XDMFFile(mesh.comm, dimensional_concentration_xdmfs[i], "w") as file:
        file.write_mesh(mesh)
        file.write_function(concentration_function)
