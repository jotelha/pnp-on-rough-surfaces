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

xscale = 0.05
x_offset_factor = 1.2
y_offset = 2
z_offset = 15
contour_label_x_offset = - 4.5
contour_label_y_offset = 0.08

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
import vtk  # needed for rendering Latex in pyvista plots correctly
            # see https://github.com/pyvista/pyvista/discussions/2928#discussioncomment-5229758
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

# clip in y direction
# grid = grid.clip(normal=(0, 1, 0), origin=(0, 7, 0))

bounds = grid.bounds
logger.info("grid bounds: %s", bounds)

# Calculate the lower edge center in the XY plane (Z = 0)
focal_point = [
    (bounds[0] + bounds[1]) / 2 * x_offset_factor,  # X center
    bounds[2] + y_offset,                    # Y at the lower bound
    (bounds[4] + bounds[5]) / 2   # Z center (to stay in the middle Z plane)
]

logger.info("focal point: %s", focal_point)

# clip in x direction
# (bounds[0] + bounds[1]) / 2 * xscale * x_offset_factor,
upper_x_clip = focal_point[0] + 80
lower_x_clip = focal_point[0] - 80

upper_y_clip = focal_point[1] - y_offset + 5
lower_y_clip = focal_point[1] - y_offset - 1
# print(lower_x_clip)
# grid = grid.clip(normal=(1, 0, 0), origin=(upper_x_clip, 0, 0))
# grid = grid.clip(normal=(1, 0, 0), origin=(lower_x_clip, 0, 0))
clipping_box = [lower_x_clip, upper_x_clip, lower_y_clip, upper_y_clip, -1, 1]

logger.info("clipping box: %s", clipping_box)
grid = grid.clip_box(clipping_box, invert=False)

bounds = grid.bounds
logger.info("clipped grid bounds: %s", bounds)

# recompute focal point
focal_point = [
    (bounds[0] + bounds[1]) / 2 * xscale,  # X center
    bounds[2] + y_offset,                    # Y at the lower bound
    (bounds[4] + bounds[5]) / 2   # Z center (to stay in the middle Z plane)
]
logger.info("recomputed focal point: %s", focal_point)

# contour_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
contour_values = np.geomspace(0.01*potential_bias, 0.9*potential_bias, num=8)
contours = grid.contour(isosurfaces=contour_values)
levels = contours.split_bodies()

# Label the contours
pts = []

# Position the camera to look straight down on the XY plane
# Keep the camera at a height above the grid
camera_position = [
    focal_point[0],            # X position
    focal_point[1],            # Y position
    focal_point[2] + z_offset  # Z position (height above the plane)
]
logger.info("camera position: %s", focal_point)

for level, value in zip(levels, contour_values[::-1]):
    pt = np.mean(level.points, axis=0)
    pt[0] = camera_position[0] + contour_label_x_offset
    pt[1] += contour_label_y_offset
    pt[2] += 0.5
    logger.info("contour %.2f: coordinates %s", value, pt)
    pts.append(pt)

contour_labels = [f'{v:.2f}' for v in contour_values[::-1]]
contour_label_coordinates = np.array(pts)

theme = pyvista.themes.DocumentProTheme()
theme.font.family = 'arial'
# theme.font.size = 30
# theme.font.label_size = 6
theme.font.color = 'black'
# theme.sow_edges = True

logger.info("Plot potential.")
plotter = pyvista.Plotter(theme=theme)

annotations = {
    potential_bias: f"{potential_bias:.2f}",
    1: "1",
    0.01: "0"
}

plotter.add_mesh(grid, show_edges=False,
                 scalar_bar_args={
                     'title': '                     potential $(U_T)$',
                     'title_font_size': 16,
                     'label_font_size': 14,
                     'n_labels': 0,
                     'fmt': '%.2f',
                     #'shadow': True,
                     # 'color': 'white',  # Color of the labels and title
                     'vertical': True,  # Orientation of the colorbar
                     'position_x': 0.77,  # Position of the colorbar in the plot (x-coordinates)
                     'position_y': 0.2,  # Position of the colorbar in the plot (y-coordinates)
                     'width': 0.08,      # Width of the colorbar
                     'height': 0.5,      # Height of the colorbar
                 },
                 annotations=annotations)

scalar_bar = plotter.scalar_bar
scalar_bar.SetTextPosition(-1)
title_text_property = scalar_bar.GetTitleTextProperty()
title_text_property.SetJustificationToRight()
title_text_property.SetLineOffset(10)

logger.info("Add %d labels at %d points", len(contour_labels), len(contour_label_coordinates))
#plotter.add_point_labels(
#    contour_label_coordinates, contour_labels,
#    text_color='w', shape=None)
plotter.add_mesh(contours, line_width=3, render_lines_as_tubes=True, color='w')
plotter.set_scale(xscale=xscale)
plotter.view_xy()

logger.info("xscale: %g", xscale)
logger.info("x_offset_factor: %g", x_offset_factor)
logger.info("y_offset: %g", y_offset)
logger.info("z_offset: %g", z_offset)

# Adjust camera focal point to the lower edge center
plotter.camera.focal_point = focal_point
plotter.camera.position = camera_position

# # Customize the axes
# axes = pyvista.Axes()
#
# # Hide the z-axis
# axes.actor.z_axis_visibility = False  # Hide the z-axis
#
# # Customize tick visibility for x and y axes if needed
# axes.actor.x_axis_tick_visibility = True
# axes.actor.y_axis_tick_visibility = True
#
# # Customize tick label properties
# axes.actor.x_axis_tick_labels_text_property.font_size = 12
# axes.actor.y_axis_tick_labels_text_property.font_size = 12
#
# # Add customized axes to the plotter
# plotter.add_actor(axes)
#

scaled_bounds = (
    bounds[0] * xscale,
    bounds[1] * xscale,
    bounds[2],
    bounds[3],
    0,
    0)

#plotter.show_bounds(
#    bounds=scaled_bounds,
#    xlabel='x', ylabel='y',
#    location='origin',
#    padding=0.01,
#    ticks='outside',
#    minor_ticks=True,
#    show_zaxis=False,
#    show_zlabels=False,
#    yticks=[0,1,2,3,4,5])

pointa = [bounds[0] * xscale, bounds[2] - 0.3, 0]
pointb = [bounds[1] * xscale, bounds[2] - 0.3, 0]
xruler = plotter.add_ruler(pointa, pointb,
                           title='x ($\lambda_\mathrm{D}$)', label_format='%.0f',
                           font_size_factor=0.8,
                           label_size_factor=0.7)

title_text_property = xruler.GetTitleTextProperty()
title_text_property.BoldOff()
title_text_property.ItalicOff()

label_text_property = xruler.GetLabelTextProperty()
label_text_property.BoldOff()
label_text_property.ItalicOff()

xruler.SetRange(bounds[0], bounds[1])

pointa = [bounds[0] * xscale - 0.3, 0, 0]
pointb = [bounds[0] * xscale - 0.3, 5, 0]
yruler = plotter.add_ruler(pointb, pointa,
                          title='y $(\lambda_\mathrm{D})$', label_format='%.0f',
                          flip_range=True,
                          font_size_factor=0.8,
                          label_size_factor=0.7)
title_text_property = yruler.GetTitleTextProperty()
title_text_property.BoldOff()
title_text_property.ItalicOff()

label_text_property = yruler.GetLabelTextProperty()
label_text_property.BoldOff()
label_text_property.ItalicOff()

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
    plotter.enable_parallel_projection()

    plotter.add_mesh(grid, show_edges=False)
    # plotter.add_mesh(contours, line_width=2, render_lines_as_tubes=True, color='r')
    plotter.view_xy()

    xscale = 0.05

    plotter.set_scale(xscale=xscale)

    bounds = grid.bounds

    # Calculate the lower edge center in the XY plane (Z = 0)
    focal_point = [
        (bounds[0] + bounds[1]) / 2 * xscale * 1.2,  # X center
        bounds[2] + 2,  # Y at the lower bound
        (bounds[4] + bounds[5]) / 2  # Z center (to stay in the middle Z plane)
    ]

    # Adjust camera focal point to the lower edge center
    plotter.camera.focal_point = focal_point

    # Position the camera to look straight down on the XY plane
    # Keep the camera at a height above the grid
    camera_position = [
        focal_point[0],  # X position
        focal_point[1],  # Y position
        focal_point[2] + 15  # Z position (height above the plane)
    ]

    plotter.camera.position = camera_position

    plotter.show_axes()
    plotter.show()
    logger.info("Dump concentration %i plot to %s.", i, output[i])
    plotter.screenshot(output[i])