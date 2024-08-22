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

scale_factor = 2

xscale = 0.05
# x_offset_factor = 1.1
x_focal_point = 275
y_offset = 2
z_offset = 10
contour_label_x_offset = - 3.6
contour_label_y_offset = 0.12

upper_y_clip_offset = 2
lower_y_clip_offset = -1

x_clip_width = 50

contour_width = 6

x_tick_spacing = 25

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

import scipy.constants as sc

gas_constant = sc.value('molar gas constant')
faraday_constant = sc.value('Faraday constant')

thermal_voltage = gas_constant * temperature / faraday_constant
potential_bias = potential_bias_SI / thermal_voltage

concentration_labels = ["$[\mathrm{H}_3\mathrm{O}^+] (c_{\mathrm{bulk}})$",
                        "$[\mathrm{OH}^-] (c_{\mathrm{bulk}})$"]

x_label = 'Length $x$ ($\lambda_\mathrm{D}$)'
y_label = 'Height $z$ ($\lambda_\mathrm{D}$)'

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

# potential
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

grid.point_data["potential"] = potential_function.x.array.real
grid.set_active_scalars("potential")

bounds = grid.bounds
logger.info("grid bounds: %s", bounds)

# Calculate the lower edge center in the XY plane (Z = 0)
focal_point = [
    #(bounds[0] + bounds[1]) / 2 * x_offset_factor,  # X center
    x_focal_point,
    bounds[2] + y_offset,                    # Y at the lower bound
    (bounds[4] + bounds[5]) / 2   # Z center (to stay in the middle Z plane)
]

logger.info("focal point: %s", focal_point)

# clip in x direction
upper_x_clip = focal_point[0] + x_clip_width/2
lower_x_clip = focal_point[0] - x_clip_width/2

upper_y_clip = focal_point[1] - y_offset + upper_y_clip_offset
lower_y_clip = focal_point[1] - y_offset + lower_y_clip_offset
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
contour_values = np.geomspace(0.02*potential_bias, 0.9*potential_bias, num=7)
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
    pt[0] = (camera_position[0] + contour_label_x_offset)/xscale
    pt[1] += contour_label_y_offset
    pt[2] += 0.1
    logger.info("contour %.2f: coordinates %s", value, pt)
    pts.append(pt)

contour_labels = [f'{v:.2f}' for v in contour_values[::-1]]
contour_label_coordinates = np.array(pts)

logger.info("Contour labels: %s", contour_labels)
logger.info("Contour label coordinates: %s", contour_label_coordinates)


theme = pyvista.themes.DocumentProTheme()
theme.font.family = 'arial'
theme.font.color = 'black'

logger.info("Plot potential.")
plotter = pyvista.Plotter(theme=theme)
plotter.image_scale = scale_factor

# plotter = pyvista.Plotter()
annotations = {
    potential_bias: f"{potential_bias:.2f}",
    1: "1",
    0.01: "0"
}

plotter.add_mesh(grid, show_edges=False,
                 scalar_bar_args={
                     'title': '                       Potential $\phi\, (U_T)$',
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
plotter.add_mesh(contours, line_width=contour_width, render_lines_as_tubes=True, color='w')
plotter.set_scale(xscale=xscale)
plotter.view_xy()

for label, position in zip(contour_labels, contour_label_coordinates):

    renderer = plotter.renderer
    camera = renderer.GetActiveCamera()

    world_coords = [position[0]*xscale, position[1], position[2], 1.0]
    logger.info(f"World coordinates: {world_coords}")

    # Get the transformation matrix from world to view coordinates
    world_to_view_transform = camera.GetCompositeProjectionTransformMatrix(renderer.GetTiledAspectRatio(), -1, 1)

    # Apply the transformation
    view_coords = [0.0, 0.0, 0.0, 0.0]  # Initialize to hold transformed coordinates
    world_to_view_transform.MultiplyPoint(world_coords, view_coords)
    logger.info(f"view coordinates: {view_coords}")

    # Perspective division to get normalized device coordinates (NDC)
    ndc_coords = [view_coords[i] / view_coords[3] for i in range(3)]
    logger.info(f"NDC coordinates: {ndc_coords}")

    # Ensure NDC coordinates are within [-1, 1] range
    clamped_ndc_coords = np.clip(ndc_coords[:2], -1, 1)
    logger.info(f"Clamped NDC coordinates: {clamped_ndc_coords}")

    # Convert NDC to viewport coordinates (pixel coordinates)
    window_size = plotter.window_size
    logger.info(f"Window size: {window_size}")

    viewport_coords = [
        (clamped_ndc_coords[0] + 1) * 0.5 * window_size[0] * scale_factor,  # X viewport
        (clamped_ndc_coords[1] + 1) * 0.5 * window_size[1] * scale_factor # Y viewport
    ]

    logger.info(f"Viewport coordinates: {viewport_coords}")

    plotter.add_text(
        label,
        position=viewport_coords,
        font_size=8,
        color='w')

logger.info("xscale: %g", xscale)
# logger.info("x_offset_factor: %g", x_offset_factor)#
logger.info("x_focal_point: %g", x_focal_point)
logger.info("y_offset: %g", y_offset)
logger.info("z_offset: %g", z_offset)

# Adjust camera focal point to the lower edge center
plotter.camera.focal_point = focal_point
plotter.camera.position = camera_position

scaled_bounds = (
    bounds[0] * xscale,
    bounds[1] * xscale,
    bounds[2],
    bounds[3],
    0,
    0)

logger.info("Scaled bounds: %s", scaled_bounds)

pointa = [bounds[0] * xscale, bounds[2] - 0.3, 0]
pointb = [bounds[1] * xscale, bounds[2] - 0.3, 0]
logger.info("x ruler from %s to %s", pointa, pointb)
xruler = plotter.add_ruler(pointa, pointb,
                           title=x_label,
                           label_format='%.0f',
                           font_size_factor=0.8,
                           label_size_factor=0.7)
# keyword scale does not exist in this version

xruler_range = xruler.GetRange()
logger.info("xruler range: %s", xruler_range)
x_range = np.array([bounds[0], bounds[1]])
logger.info("x range: %s", x_range)
xruler_scale_factor = (xruler_range[1] - xruler_range[0])/(x_range[1]-x_range[0])
logger.info("xruler range / x range scale factor: %g", xruler_scale_factor)
xruler.SetRange(bounds[0], bounds[1])
new_xruler_range = xruler.GetRange()

number_of_ticks = int(np.round(x_range[1]-x_range[0])/x_tick_spacing)+1
logger.info("number of ticks: %s", number_of_ticks)
logger.info("new  xruler range: %s", new_xruler_range)

xruler.AdjustLabelsOff()
# xruler.SetNumberOfMinorTicks(number_of_minor_ticks)
xruler.SetNumberOfLabels(number_of_ticks)

title_text_property = xruler.GetTitleTextProperty()
title_text_property.BoldOff()
title_text_property.ItalicOff()

label_text_property = xruler.GetLabelTextProperty()
label_text_property.BoldOff()
label_text_property.ItalicOff()

pointa = [bounds[0] * xscale - 0.3, 0, 0]
pointb = [bounds[0] * xscale - 0.3, upper_y_clip_offset, 0]
logger.info("y ruler from %s to %s", pointa, pointb)
yruler = plotter.add_ruler(pointb, pointa,
                           title=y_label, label_format='%.0f',
                           flip_range=True,
                           font_size_factor=0.8,
                           label_size_factor=0.7)
title_text_property = yruler.GetTitleTextProperty()
title_text_property.BoldOff()
title_text_property.ItalicOff()
# title_text_property.SetOrientation(90)

label_text_property = yruler.GetLabelTextProperty()
label_text_property.BoldOff()
label_text_property.ItalicOff()

# plotter.show()
logger.info("Dump potential plot to %s.", output.potential_png)
plotter.screenshot(output.potential_png)

# concentrations

# xscale = 0.02
# x_offset_factor = 0.93
# y_offset = 2
# z_offset = 15
contour_label_x_offset = - 3.6
contour_label_y_offset = 0.0

# x_clip_width = 400

for i, concentration_function in enumerate(concentration_functions):
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["concentration"] = concentration_function.x.array.real
    grid.set_active_scalars("concentration")

    bounds = grid.bounds
    logger.info("grid bounds: %s", bounds)

    # Calculate the lower edge center in the XY plane (Z = 0)
    focal_point = [
        # (bounds[0] + bounds[1]) / 2 * x_offset_factor,  # X center
        x_focal_point,
        bounds[2] + y_offset,  # Y at the lower bound
        (bounds[4] + bounds[5]) / 2  # Z center (to stay in the middle Z plane)
    ]

    logger.info("focal point: %s", focal_point)

    # clip in x direction
    upper_x_clip = focal_point[0] + x_clip_width / 2
    lower_x_clip = focal_point[0] - x_clip_width / 2

    # upper_y_clip = focal_point[1] - y_offset + upper_y_clip_offset
    upper_y_clip = upper_y_clip_offset
    # lower_y_clip = focal_point[1] - y_offset + lower_y_clip_offset
    lower_y_clip = lower_y_clip_offset
    clipping_box = [lower_x_clip, upper_x_clip, lower_y_clip, upper_y_clip, -1, 1]

    logger.info("clipping box: %s", clipping_box)
    grid = grid.clip_box(clipping_box, invert=False)

    bounds = grid.bounds
    logger.info("clipped grid bounds: %s", bounds)

    # recompute focal point
    focal_point = [
        (bounds[0] + bounds[1]) / 2 * xscale,  # X center
        bounds[2] + y_offset,  # Y at the lower bound
        (bounds[4] + bounds[5]) / 2  # Z center (to stay in the middle Z plane)
    ]
    logger.info("recomputed focal point: %s", focal_point)

    cmin = np.min(concentration_function.x.array.real)
    cmax = np.max(concentration_function.x.array.real)
    cspan = cmax - cmin

    logger.info("Minimum concentration: %g", cmin)
    logger.info("Maximum concentration: %g", cmax)
    contour_values = np.geomspace(cmin+ 0.02 * cspan, cmin + 0.9 * cspan, num=10)
    contours = grid.contour(isosurfaces=contour_values)
    levels = contours.split_bodies()

    # Label the contours
    pts = []

    # Position the camera to look straight down on the XY plane
    # Keep the camera at a height above the grid
    camera_position = [
        focal_point[0],  # X position
        focal_point[1],  # Y position
        focal_point[2] + z_offset  # Z position (height above the plane)
    ]
    logger.info("camera position: %s", focal_point)

    if contour_values[0] > 1: # weird adjustment for ordering
         contour_values = contour_values[::-1]
    for level, value in zip(levels, contour_values):
        pt = np.mean(level.points, axis=0)
        pt[0] = (camera_position[0] + contour_label_x_offset) / xscale
        pt[1] += contour_label_y_offset
        pt[2] += 0.1
        logger.info("contour %.2f: coordinates %s", value, pt)
        pts.append(pt)

    contour_labels = [f'{v:.2f}' for v in contour_values]
    contour_label_coordinates = np.array(pts)

    logger.info("Contour labels: %s", contour_labels)
    logger.info("Contour label coordinates: %s", contour_label_coordinates)

    theme = pyvista.themes.DocumentProTheme()
    theme.font.family = 'arial'
    theme.font.color = 'black'

    logger.info("Plot potential.")
    plotter = pyvista.Plotter(theme=theme)
    plotter.image_scale = 2
    # plotter = pyvista.Plotter()
    annotations = {
        cmin+0.01*cspan: f"{cmin:.2f}",
        cmin+0.5*cspan: f"{cmin+0.5*cspan:.2f}",
        cmax-0.01*cspan: f"{cmax:.2f}"
    }

    plotter.add_mesh(grid, show_edges=False,
                     scalar_bar_args={
                         'title': f'                     {concentration_labels[i]}',
                         'title_font_size': 16,
                         'label_font_size': 14,
                         'n_labels': 0,
                         'fmt': '%.2f',
                         # 'shadow': True,
                         # 'color': 'white',  # Color of the labels and title
                         'vertical': True,  # Orientation of the colorbar
                         'position_x': 0.77,  # Position of the colorbar in the plot (x-coordinates)
                         'position_y': 0.2,  # Position of the colorbar in the plot (y-coordinates)
                         'width': 0.08,  # Width of the colorbar
                         'height': 0.5,  # Height of the colorbar
                     },
                     annotations=annotations)

    scalar_bar = plotter.scalar_bar
    scalar_bar.SetTextPosition(-1)
    title_text_property = scalar_bar.GetTitleTextProperty()
    title_text_property.SetJustificationToRight()
    title_text_property.SetLineOffset(10)

    logger.info("Add %d labels at %d points", len(contour_labels), len(contour_label_coordinates))
    plotter.add_mesh(contours, line_width=contour_width, render_lines_as_tubes=True, color='w')
    plotter.set_scale(xscale=xscale)
    plotter.view_xy()

    for label, position in zip(contour_labels, contour_label_coordinates):
        renderer = plotter.renderer
        camera = renderer.GetActiveCamera()

        world_coords = [(x_focal_point+x_clip_width/2)*xscale, position[1], position[2], 1.0]
        logger.info(f"World coordinates: {world_coords}")

        # Get the transformation matrix from world to view coordinates
        world_to_view_transform = camera.GetCompositeProjectionTransformMatrix(renderer.GetTiledAspectRatio(), -1, 1)

        # Apply the transformation
        view_coords = [0.0, 0.0, 0.0, 0.0]  # Initialize to hold transformed coordinates
        world_to_view_transform.MultiplyPoint(world_coords, view_coords)
        logger.info(f"view coordinates: {view_coords}")

        # Perspective division to get normalized device coordinates (NDC)
        ndc_coords = [view_coords[i] / view_coords[3] for i in range(3)]
        logger.info(f"NDC coordinates: {ndc_coords}")

        # Ensure NDC coordinates are within [-1, 1] range
        clamped_ndc_coords = np.clip(ndc_coords[:2], -1, 1)
        logger.info(f"Clamped NDC coordinates: {clamped_ndc_coords}")

        # Convert NDC to viewport coordinates (pixel coordinates)
        window_size = plotter.window_size
        logger.info(f"Window size: {window_size}")

        viewport_coords = [
            (clamped_ndc_coords[0] + 1) * 0.5 * window_size[0] * scale_factor,  # X viewport
            (clamped_ndc_coords[1] + 1) * 0.5 * window_size[1] * scale_factor # Y viewport
        ]

        logger.info(f"Viewport coordinates: {viewport_coords}")

        plotter.add_text(
            label,
            position=viewport_coords,
            font_size=6,
            color='k')

    logger.info("xscale: %g", xscale)
    # logger.info("x_offset_factor: %g", x_offset_factor)
    logger.info("x_focal_point: %g", x_focal_point)
    logger.info("y_offset: %g", y_offset)
    logger.info("z_offset: %g", z_offset)

    # Adjust camera focal point to the lower edge center
    plotter.camera.focal_point = focal_point
    plotter.camera.position = camera_position

    scaled_bounds = (
        bounds[0] * xscale,
        bounds[1] * xscale,
        bounds[2],
        bounds[3],
        0,
        0)

    logger.info("Scaled bounds: %s", scaled_bounds)

    pointa = [bounds[0] * xscale, bounds[2] - 0.3, 0]
    pointb = [bounds[1] * xscale, bounds[2] - 0.3, 0]
    logger.info("x ruler from %s to %s", pointa, pointb)
    xruler = plotter.add_ruler(pointa, pointb,
                               title=x_label,
                               label_format='%.0f',
                               font_size_factor=0.8,
                               label_size_factor=0.7)

    xruler_range = xruler.GetRange()
    logger.info("xruler range: %s", xruler_range)
    x_range = np.array([bounds[0], bounds[1]])
    logger.info("x range: %s", x_range)
    xruler_scale_factor = (xruler_range[1] - xruler_range[0])/(x_range[1]-x_range[0])
    logger.info("xruler range / x range scale factor: %g", xruler_scale_factor)

    xruler.SetRange(x_range[0], x_range[1])

    title_text_property = xruler.GetTitleTextProperty()
    title_text_property.BoldOff()
    title_text_property.ItalicOff()

    label_text_property = xruler.GetLabelTextProperty()
    label_text_property.BoldOff()
    label_text_property.ItalicOff()

    xruler_range = xruler.GetRange()
    logger.info("xruler range: %s", xruler_range)
    x_range = np.array([bounds[0], bounds[1]])
    logger.info("x range: %s", x_range)
    xruler_scale_factor = (xruler_range[1] - xruler_range[0]) / (x_range[1] - x_range[0])
    logger.info("xruler range / x range scale factor: %g", xruler_scale_factor)
    xruler.SetRange(bounds[0], bounds[1])
    new_xruler_range = xruler.GetRange()

    number_of_ticks = int(np.round(x_range[1] - x_range[0]) / x_tick_spacing) + 1
    logger.info("number of ticks: %s", number_of_ticks)
    logger.info("new  xruler range: %s", new_xruler_range)

    xruler.AdjustLabelsOff()
    xruler.SetNumberOfLabels(number_of_ticks)

    pointa = [bounds[0] * xscale - 0.3, 0, 0]
    pointb = [bounds[0] * xscale - 0.3, upper_y_clip_offset, 0]
    logger.info("y ruler from %s to %s", pointa, pointb)
    yruler = plotter.add_ruler(pointb, pointa,
                               title=y_label,
                               label_format='%.0f',
                               flip_range=True,
                               font_size_factor=0.8,
                               label_size_factor=0.7)

    yruler.AdjustLabelsOff()
    yruler.SetNumberOfLabels(int(upper_y_clip_offset)+1)

    title_text_property = yruler.GetTitleTextProperty()
    title_text_property.BoldOff()
    title_text_property.ItalicOff()

    label_text_property = yruler.GetLabelTextProperty()
    label_text_property.BoldOff()
    label_text_property.ItalicOff()

    logger.info("Dump concentration %i plot to %s.", i, output[i])
    plotter.screenshot(output[i])