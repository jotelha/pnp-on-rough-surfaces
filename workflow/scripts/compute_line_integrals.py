input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

number_of_species = config["number_of_species"]
height_normalized = config["height_normalized"]

mesh_msh = input.mesh_msh
checkpoint_bp = input.solution_checkpoint_bp

integrals_csv = output.integrals_csv

import os.path
import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import basix
import dolfinx
import numpy as np
import adios4dolfinx

import scipy.constants as sc

from mpi4py import MPI

import gmsh

# get boundary

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
# coords_3d *= l_unit

# read dimensional solution
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

tol = 1e-6
y_max = height_normalized
N_points = 5001

logger.info("tol: %g", tol)
logger.info("y_max: %g", y_max)

concentration_integrals = [[] for _ in range(number_of_species)]
excess_concentration_integrals = [[] for _ in range(number_of_species)]

bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

# os.makedirs(concentration_profile_directory, exist_ok=True)

for j, (x, y_min) in enumerate(zip(coords_3d[:,0], coords_3d[:,1])):
    y_grid = np.linspace(y_min + tol, y_max - tol, N_points)

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
        excess_concentration_profile = concentration_profile - 1  # reference concentration is always 1

        logger.info("concentration_profile.shape: %s", concentration_profile.shape)

        concentration_integral = np.trapz(y=concentration_profile, x=y_grid)
        excess_concentration_integral = np.trapz(y=excess_concentration_profile, x=y_grid)

        logger.info("concentration_integral: %s", concentration_integral)
        logger.info("concentration_integral.shape: %s", concentration_integral.shape)

        logger.info("excess_concentration_integral: %s", excess_concentration_integral)
        logger.info("excess_concentration_integral.shape: %s", excess_concentration_integral.shape)

        # concentration_profiles.append(concentration_profile.flatten())
        # excess_concentration_profiles.append(excess_concentration_profile.flatten())

        concentration_integrals[i].append(concentration_integral)
        excess_concentration_integrals[i].append(excess_concentration_integral)

for i in range(number_of_species):
    concentration_integrals[i] = np.array(concentration_integrals[i]).flatten()
    excess_concentration_integrals[i] = np.array(excess_concentration_integrals[i]).flatten()
    logger.info("concentration_integrals[%d].shape: %s", i, concentration_integrals[i].shape)

coords_and_integrals = np.vstack([coords_3d[:,0], coords_3d[:,1],
                                  *concentration_integrals, *excess_concentration_integrals]).T

header = ','.join(['x','y',
              *[f'concentration_integral_{i}' for i in range(number_of_species)],
              *[f'excess_concentration_integral_{i}' for i in range(number_of_species)]])

with open(integrals_csv, 'w') as f:
    f.write(header + "\n")
    np.savetxt(f, coords_and_integrals, delimiter=",")