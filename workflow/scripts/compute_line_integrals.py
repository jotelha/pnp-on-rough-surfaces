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
height_normalized = config["height_normalized"]

mesh_msh = input.mesh_msh
dimensional_solution_checkpoint_bp = input.dimensional_solution_checkpoint_bp

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import basix
import dolfinx
import numpy as np
import ufl
import adios4dolfinx

import scipy.constants as sc

from mpi4py import MPI

import gmsh

# get boundary

# helper functions
def ionic_strength(z, c):
    """Compute a system's ionic strength from charges and concentrations.

    Returns
    -------
    ionic_strength : float
        ionic strength ( 1/2 * sum(z_i^2*c_i) )
        [concentration unit, i.e. mol m^-3]
    """
    return 0.5*np.sum(np.square(z) * c)

vacuum_permittivity = sc.epsilon_0
gas_constant = sc.value('molar gas constant')
faraday_constant = sc.value('Faraday constant')


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

gmsh.initialize()

# Open the mesh file
gmsh.open(mesh_msh)

# get all line segments belonging to tag 4, the rough edge
# entities = gmsh.model.getEntitiesForPhysicalGroup(1, 4)

# get nodes on rough edge
nodes, coords = gmsh.model.mesh.getNodesForPhysicalGroup(1, 4)

coords_3d = np.reshape(coords, (len(nodes),3))

sorted_indices = np.argsort(coords_3d[:,0])

coords_3d = coords_3d[sorted_indices,:]

logger.info("coords_3d.shape: %s", coords_3d.shape)

# scale coords
#
coords_3d *= l_unit

# read dimensional solution
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

tol = l_unit*1e-6
y_max = 7*l_unit
N_points = 1001

logger.info("Debye length: %g", debye_length)
logger.info("l_unit: %g", l_unit)
logger.info("tol: %g", tol)
logger.info("y_max: %g", y_max)

concentration_integrals = [[] for _ in range(number_of_species)]
concentration_values = [[] for _ in range(number_of_species)]

bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

for x, y_min in zip(coords_3d[:,0], coords_3d[:,1]):
    y_grid = np.linspace(y_min + tol, y_max - tol, N_points)

    #if x == 0:
    #    x = tol

    points = np.zeros((3, N_points))
    points[0, :] = x
    points[1, :] = y_grid

    logger.info("points.shape: %s", points.shape)

    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
        else:
            logger.warning("Point %d: %s not in domain", i, point)

    points_on_proc = np.array(points_on_proc, dtype=np.float64)

    logger.info("points_on_proc.shape: %s", points_on_proc.shape)

    for i in range(number_of_species):
        concentration_profile = concentration_functions[i].eval(points_on_proc, cells).T
        concentration_values[i].append(concentration_profile)
        concentration_integrals[i].append(np.trapz(y=concentration_profile, x=y_grid))

for i in range(number_of_species):
    np.savetxt(output[i], np.array(concentration_values[i]))

coords_and_integrals = np.hstack([coords_3d[:,0], coords_3d[:,1], *concentration_integrals])
np.savetxt(output.integrals_csv, coords_and_integrals, header='x, y, concentration integrals')