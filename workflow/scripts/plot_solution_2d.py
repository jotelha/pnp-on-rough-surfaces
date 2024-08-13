import numpy as np

input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

number_of_species = config["number_of_species"]
temperature = config["temperature"]
potential_bias_SI = config["potential_bias"]
checkpoint_bp = input.interpolated_solution_checkpoint_bp

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import scipy.constants as sc

gas_constant = sc.value('molar gas constant')
faraday_constant = sc.value('Faraday constant')

thermal_voltage = gas_constant * temperature / faraday_constant
potential_bias = potential_bias_SI / thermal_voltage

import basix
import dolfinx
import adios4dolfinx
import pyvista

from dolfinx import plot

from mpi4py import MPI

logger.info("Read mesh from file '%s'.", checkpoint_bp)
mesh = adios4dolfinx.read_mesh(checkpoint_bp,
                               comm=MPI.COMM_WORLD,
                               engine="BP4",
                               ghost_mode=dolfinx.mesh.GhostMode.none)

single_element_CG1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
scalar_function_space_CG1 = dolfinx.fem.functionspace(mesh, single_element_CG1)

potential_function = dolfinx.fem.Function(scalar_function_space_CG1,
                                          dtype=dolfinx.default_scalar_type)
logger.info("Read potential from file '%s'.", checkpoint_bp)
adios4dolfinx.read_function(checkpoint_bp,
                            potential_function,
                            name="solution_function_0")

concentration_functions = []
for i in range(number_of_species):
    concentration_function = dolfinx.fem.Function(
        scalar_function_space_CG1, dtype=dolfinx.default_scalar_type)
    logger.info("Read concentration %d from file '%s'.", i, checkpoint_bp)
    adios4dolfinx.read_function(checkpoint_bp,
                                concentration_function,
                                name=f"solution_function_{i+1}")
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

grid = grid.clip(normal=(0, 1, 0), origin=(0, 7, 0))

# contour_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
contour_values = np.geomspace(0.01*potential_bias, 0.9*potential_bias, num=8)
contours = grid.contour(isosurfaces=contour_values)
levels = contours.split_bodies()
print(len(levels))
# Label the contours
pts = []
# values = []
xscale = 0.05
x_offset_factor = 1.2
y_offset = 2
z_offset = 15
contour_label_x_offset = - 5
contour_label_y_offset = 0.08

bounds = grid.bounds

# Calculate the lower edge center in the XY plane (Z = 0)
lower_edge_center = [
    (bounds[0] + bounds[1]) / 2 * xscale * x_offset_factor,  # X center
    bounds[2] + y_offset,                    # Y at the lower bound
    (bounds[4] + bounds[5]) / 2   # Z center (to stay in the middle Z plane)
]

logger.info("focal point: %s", lower_edge_center)

# Position the camera to look straight down on the XY plane
# Keep the camera at a height above the grid
camera_position = [
    lower_edge_center[0],            # X position
    lower_edge_center[1],            # Y position
    lower_edge_center[2] + z_offset  # Z position (height above the plane)
]
logger.info("camera position: %s", lower_edge_center)

for level, value in zip(levels, contour_values[::-1]):
    pt = np.mean(level.points, axis=0)
    pt[0] = camera_position[0] + contour_label_x_offset
    pt[1] += contour_label_y_offset
    pt[2] += 0.5
    logger.info("contour %.2f: coordinates %s", value, pt)
    pts.append(pt)

    #plotter.add_point_labels(
    #    [contour_center],  # Coordinates where the label will be placed
    #     [f"{value:.2e}"],  # Label text with the contour value
    #     point_size=10,
    #     font_size=20,
    #     text_color='white',
    #     margin=0,
    #     shape=None  # No background shape around the label
    # )

contour_labels = [f'{v:.2f}' for v in contour_values[::-1]]
contour_label_coordinates = np.array(pts)

logger.info("Plot potential.")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
logger.info("Add %d labels at %d points", len(contour_labels), len(contour_label_coordinates))
plotter.add_point_labels(
    contour_label_coordinates, contour_labels,
    text_color='w', shape=None)
plotter.add_mesh(contours, line_width=3, render_lines_as_tubes=True, color='w')
plotter.view_xy()

logger.info("xscale: %g", xscale)
logger.info("x_offset_factor: %g", x_offset_factor)
logger.info("y_offset: %g", y_offset)
logger.info("z_offset: %g", z_offset)

plotter.set_scale(xscale=xscale)

# Adjust camera focal point to the lower edge center
plotter.camera.focal_point = lower_edge_center
plotter.camera.position = camera_position

# plotter.set_scale(xscale=0.1, yscale=10)
# plotter.camera.zoom(5)
plotter.show_axes()
plotter.show()
logger.info("Dump potential plot to %s.", output.potential_png)
plotter.screenshot(output.potential_png)

for i, concentration_function in enumerate(concentration_functions):
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["concentration"] = concentration_function.x.array.real
    grid.set_active_scalars("concentration")

    grid = grid.clip(normal=(0, 1, 0), origin=(0, 7, 0))

    # contour_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    # contours = grid.contour(isosurfaces=contour_values)

    logger.info("Plot concentration %d.", i)
    plotter = pyvista.Plotter()

    plotter.add_mesh(grid, show_edges=False)
    # plotter.add_mesh(contours, line_width=2, render_lines_as_tubes=True, color='r')
    plotter.view_xy()

    xscale = 0.05

    plotter.set_scale(xscale=xscale)

    bounds = grid.bounds

    # Calculate the lower edge center in the XY plane (Z = 0)
    lower_edge_center = [
        (bounds[0] + bounds[1]) / 2 * xscale * 1.2,  # X center
        bounds[2] + 2,  # Y at the lower bound
        (bounds[4] + bounds[5]) / 2  # Z center (to stay in the middle Z plane)
    ]

    # Adjust camera focal point to the lower edge center
    plotter.camera.focal_point = lower_edge_center

    # Position the camera to look straight down on the XY plane
    # Keep the camera at a height above the grid
    camera_position = [
        lower_edge_center[0],  # X position
        lower_edge_center[1],  # Y position
        lower_edge_center[2] + 15  # Z position (height above the plane)
    ]

    plotter.camera.position = camera_position

    plotter.show_axes()
    plotter.show()
    logger.info("Dump concentration %i plot to %s.", i, output[i])
    plotter.screenshot(output[i])