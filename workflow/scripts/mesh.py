input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

debye_length = config["debye_length"]
height_normalized = config["height_normalized"]

profile_csv = input.profile_csv
profile_label = wildcards.profile

profile_config = config["profiles"][profile_label]

import numpy as np
import gmsh

import logging
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def gmsh_rectangle_with_single_rough_edge(model: gmsh.model, name: str, x, y, h=2,
                                          lower_boundary_mesh_size=None,
                                          upper_boundary_mesh_size=0.5) -> gmsh.model:
    """Create a Gmsh model of a rectangle with one rough edge.

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.
        x, y: roughness profile

    Returns:
        Gmsh model with a rectangle mesh added.
        Right, upper, left and lower (rough) boundary are tagged 1, 2, 3, 4.
    """

    dx = np.mean(x[1:]-x[:-1])

    if lower_boundary_mesh_size is None:
        lower_boundary_mesh_size = 0.5*dx

    y_mean = np.mean(y)
    y_zero_aligned = y - y_mean
    x_zero_aligned = x - x[0] + dx

    x0 = 0
    x1 = x_zero_aligned[-1] + dx

    y0 = 0
    y1 = h

    logger.info("y_mean: %g", y_mean)
    logger.info("dx: %g", dx)
    logger.info("x0, x1: %f, %f", x0, x1)

    model.add(name)
    model.setCurrent(name)

    p1 = model.geo.addPoint(x1, y0, 0, meshSize=lower_boundary_mesh_size)
    p2 = model.geo.addPoint(x1, y1, 0, meshSize=upper_boundary_mesh_size)
    l1 = model.geo.addLine(p1, p2)
    
    p3 = model.geo.addPoint(x0, y1, 0, meshSize=upper_boundary_mesh_size)
    l2 = model.geo.addLine(p2, p3)
    
    p4 = model.geo.addPoint(x0, y0, 0, meshSize=lower_boundary_mesh_size)
    l3 = model.geo.addLine(p3, p4)

    rough_edge = []
    p_previous = p4
    for x_next, y_next in zip(x_zero_aligned, y_zero_aligned):
        p_next = model.geo.addPoint(x_next, y_next, 0, meshSize=lower_boundary_mesh_size)
        l_next = model.geo.addLine(p_previous, p_next)
        p_previous = p_next
        rough_edge.append(l_next)

    l4 = model.geo.addLine(p_next, p1)
    rough_edge.append(l4)

    lines = [l1,l2,l3,*rough_edge]

    loop = model.geo.addCurveLoop(lines)
    surface = model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()
    
    # Define physical groups (optional)
    model.addPhysicalGroup(1, [l1], tag=1, name="Right boundary")
    model.addPhysicalGroup(1, [l2], tag=2, name="Upper boundary")
    model.addPhysicalGroup(1, [l3], tag=3, name="Left boundary")
    model.addPhysicalGroup(1, rough_edge, tag=4, name="Rough boundary")
    model.addPhysicalGroup(2, [surface], tag=5, name="Domain")

    return model


x_dimensional, y_dimensional = np.loadtxt(profile_csv,
                          skiprows=profile_config["skiprows"],
                          delimiter=profile_config["delimiter"],
                          usecols=profile_config["usecols"],
                          unpack=profile_config["unpack"],
                          max_rows=profile_config["max_rows"])

x_normalized = x_dimensional * profile_config["xscale"] / debye_length
y_normalized = y_dimensional * profile_config["yscale"] / debye_length

gmsh.initialize()
gmsh.clear()
model = gmsh.model()

model = gmsh_rectangle_with_single_rough_edge(
    model, profile_label, x_normalized, y_normalized, h=height_normalized,
    lower_boundary_mesh_size=0.1, upper_boundary_mesh_size=0.5)

gmsh.write(output.geometry_geo)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.0)

gmsh.model.mesh.generate(2)

gmsh.write(output.mesh_msh)