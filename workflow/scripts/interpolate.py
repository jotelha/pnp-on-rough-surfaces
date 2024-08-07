input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

number_of_species = config["number_of_species"]

solution_checkpoint_bp = input.solution_checkpoint_bp
interpolated_solution_checkpoint_bp = output.interpolated_solution_checkpoint_bp

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import basix
import dolfinx
import adios4dolfinx
import numpy as np

from mpi4py import MPI

logger.info("Read mesh from file '%s'.", solution_checkpoint_bp)
mesh = adios4dolfinx.read_mesh(solution_checkpoint_bp,
                               comm=MPI.COMM_WORLD,
                               engine="BP4",
                               ghost_mode=dolfinx.mesh.GhostMode.none)

meshtags = {}
for i in range(mesh.topology.dim + 1):
    meshtags[i] = adios4dolfinx.read_meshtags(filename=solution_checkpoint_bp,
                                              mesh=mesh, meshtag_name=f"meshtags_{i}")

facet_markers = adios4dolfinx.read_meshtags(filename=solution_checkpoint_bp,
                                            mesh=mesh, meshtag_name="facet_markers")
cell_markers = adios4dolfinx.read_meshtags(filename=solution_checkpoint_bp,
                                           mesh=mesh, meshtag_name="cell_markers")

P = basix.ufl.element('Lagrange', mesh.basix_cell(), 3)
elements = [P] * (1+number_of_species)
H = basix.ufl.mixed_element(elements)
W = dolfinx.fem.functionspace(mesh, H)

w = dolfinx.fem.Function(W)

logger.info("Read function from file '%s'.", solution_checkpoint_bp)
adios4dolfinx.read_function(solution_checkpoint_bp, w, name="solution")

solution_functions = w.split()

single_element_CG1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
scalar_function_space_CG1 = dolfinx.fem.functionspace(mesh, single_element_CG1)
interpolated_scalar_function = dolfinx.fem.Function(scalar_function_space_CG1, dtype=dolfinx.default_scalar_type)

# elements_CG1 = [single_element_CG1] * (1+number_of_species)
# mixed_element_CG1 = basix.ufl.mixed_element(elements)
# vector_function_space_CG1 = dolfinx.fem.functionspace(mesh, mixed_element_CG1)
# interpolated_vector_function = dolfinx.fem.Function(vector_function_space_CG1, dtype=dolfinx.default_scalar_type)

adios4dolfinx.write_mesh(interpolated_solution_checkpoint_bp, mesh, engine="BP4")

for i, solution_function in enumerate(solution_functions):
    logger.info("Interpolate function %d.", i)
    interpolated_scalar_function.interpolate(solution_function)

    logger.info("Write interpolated function %d to file '%s'.", i, solution_checkpoint_bp)
    adios4dolfinx.write_function(filename=interpolated_solution_checkpoint_bp,
                                 u=interpolated_scalar_function,
                                 name=f"solution_function_{i}")

    # interpolated_vector_function.sub(i).interpolate(interpolated_scalar_function)


for i in range(mesh.topology.dim + 1):
    e_map = mesh.topology.index_map(i)
    entities = np.arange(e_map.size_local, dtype=np.int32)

    # Associate each local index with its global index
    values = np.arange(e_map.size_local, dtype=np.int32) + e_map.local_range[0]
    meshtags[i] = dolfinx.mesh.meshtags(mesh, i, entities, values)

for i, tag in meshtags.items():
    adios4dolfinx.write_meshtags(filename=interpolated_solution_checkpoint_bp,
                                 mesh=mesh, meshtags=tag, meshtag_name=f"meshtags_{i}")

adios4dolfinx.write_meshtags(filename=interpolated_solution_checkpoint_bp,
                             mesh=mesh, meshtags=facet_markers, meshtag_name="facet_markers")
adios4dolfinx.write_meshtags(filename=interpolated_solution_checkpoint_bp,
                             mesh=mesh, meshtags=cell_markers, meshtag_name="cell_markers")