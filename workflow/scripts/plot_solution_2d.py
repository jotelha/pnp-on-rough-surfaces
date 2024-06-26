input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

number_of_species = config["number_of_species"]

dimensional_solution_checkpoint_bp = input.dimensional_solution_checkpoint_bp

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import basix
import dolfinx
import adios4dolfinx
import pyvista

from dolfinx import plot

from mpi4py import MPI

logger.info("Read mesh from file '%s'.", dimensional_solution_checkpoint_bp)
mesh = adios4dolfinx.read_mesh(dimensional_solution_checkpoint_bp,
                               comm=MPI.COMM_WORLD,
                               engine="BP4",
                               ghost_mode=dolfinx.mesh.GhostMode.none)

single_element_CG1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
scalar_function_space_CG1 = dolfinx.fem.functionspace(mesh, single_element_CG1)

potential_function = dolfinx.fem.Function(scalar_function_space_CG1,
                                          dtype=dolfinx.default_scalar_type)
logger.info("Read potential from file '%s'.", dimensional_solution_checkpoint_bp)
adios4dolfinx.read_function(dimensional_solution_checkpoint_bp,
                            potential_function,
                            name="potential")

concentration_functions = []
for i in range(number_of_species):
    concentration_function = dolfinx.fem.Function(
        scalar_function_space_CG1, dtype=dolfinx.default_scalar_type)
    logger.info("Read concentration %d from file '%s'.", i, dimensional_solution_checkpoint_bp)
    adios4dolfinx.read_function(dimensional_solution_checkpoint_bp,
                                concentration_function,
                                name=f"concentration_{i}")
    concentration_functions.append(concentration_function)

# plot
logger.info("Start pyvista.")
# if pyvista.OFF_SCREEN:
pyvista.start_xvfb(wait=0.1)
logger.info("Create plot grid.")
topology, cell_types, geometry = plot.vtk_mesh(mesh)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

grid.point_data["potential"] = potential_function.x.array.real
grid.set_active_scalars("potential")

logger.info("Plot potential.")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

plotter.camera.zoom(20)
plotter.show()
logger.info("Dump potential plot to %s.", output.potential_png)
plotter.screenshot(output.potential_png)

for i, concentration_function in enumerate(concentration_functions):
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["concentration"] = concentration_function.x.array.real
    grid.set_active_scalars("concentration")
    logger.info("Plot concentration %d.", i)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=False)
    plotter.view_xy()
    plotter.camera.zoom(20)
    plotter.show()
    logger.info("Dump concentration %i plot to %s.", i, output[i])
    plotter.screenshot(output[i])