# %% [markdown]
# # Poisson-Nernst-Planck systems at rough interfaces

# %% [markdown]
# Liu Chenxu carried out experiments at -0.2, 0., 0.2, 0.4, 0.6, 0.7 0.8, 1, and 1.2 V. We first look at a subset of those spaced 0.2V at the inert electrode in 1D:

# %% [markdown]
# Electrolyte has been deionized water with resistivity of 18.2 MΩ·cm.

# %% [markdown]
# We assume H3O+ and OH- concentrations of $1.0\cdot 10^{-7} \mathrm{mol}\,\mathrm{L}^-{1} = 1.0\cdot 10^{-4} \mathrm{mol}\,\mathrm{m}^-{3}$ or mM.
#
# See https://www.quora.com/If-pure-distilled-water-has-hydrogen-and-hydroxide-ions-why-doesnt-electrolysis-work

# %%
# u_list = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
u_list = [0.01, 0.05, 0.1, 0.2]

# %%
c = [1.0e-4, 1.0e-4]  #mM
z = [1, -1]

# %% [markdown]
# We look at the inert electrode first.

# %% [markdown]
# <a id="figure1"></a><figure>
# ![Figure 1](inertElectrode.svg)
#
# *Figure 1*: Inert electrode at the open half-space

# %% [markdown]
# As usual, we begin with preparing a few necessities.

# %%
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# %%
# basics
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

# electrochemistry basics
from matscipy.electrochemistry import debye, ionic_strength

# Poisson-Bolzmann distribution
from matscipy.electrochemistry.poisson_boltzmann_distribution import gamma, potential, concentration, charge_density

# Poisson-Nernst-Planck solver
from matscipy.electrochemistry import PoissonNernstPlanckSystem
from matscipy.electrochemistry.poisson_nernst_planck_solver_fenicsx import PoissonNernstPlanckSystemFEniCSx
from matscipy.electrochemistry.poisson_nernst_planck_solver_2d_fenicsx import PoissonNernstPlanckSystemFEniCSx2d

# 3rd party file output
import ase
import ase.io

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# %%
from mpi4py import MPI

# %%
from dolfinx.io import VTXWriter, XDMFFile

# %%
from dolfinx import default_scalar_type
from dolfinx.fem import Function, functionspace, FunctionSpace

# %%
import ufl

# %%
import basix

# %%
import gmsh

# %%
from dolfinx.io import gmshio

# %%
import dolfinx

# %%
import adios4dolfinx

# %%
import pyvista
from dolfinx import plot

# %% [markdown]
# ## Example: 1d solution at 0.2V

# %%
delta_u = 0.2

N = 200 # number of discretization grid points
L = 1.0e-5 # 10 micro m

# define desired system
pnp_1d = PoissonNernstPlanckSystemFEniCSx(
    c=c, z=z, L=L, delta_u=delta_u, N=N)

# %%
# system's Debye length in m
debye_length = pnp_1d.lambda_D

# %%
debye_length

# %%
pnp_1d.use_standard_interface_bc()

# %%
pnp_1d.solve();

# %%
x = np.linspace(0,L,100)

fig, (ax1, ax4) = plt.subplots(nrows=2, ncols=1, figsize=[16, 10])

ax1.axvline(x=pnp_1d.lambda_D/sc.nano, label='Debye Length', color='grey', linestyle=':')

ax1.plot(
    pnp_1d.grid/sc.nano, pnp_1d.potential, 
    marker='', color='tab:red', label='potential', linewidth=1, linestyle='-')

ax2 = ax1.twinx()
ax2.plot(pnp_1d.grid/sc.nano, pnp_1d.concentration[0], 
    marker='', color='tab:orange', label='H3O+', linewidth=1, linestyle='-')
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], 
    label='bulk concentration', color='grey', linewidth=2, linestyle='-.')


ax2.plot(pnp_1d.grid/sc.nano, pnp_1d.concentration[1], 
    marker='', color='tab:blue', label='OH-', linewidth=1, linestyle='-')


ax3 = ax1.twinx()
# Offset the right spine of ax3.  The ticks and label have already been
# placed on the right by twinx above.
ax3.spines["right"].set_position(("axes", 1.1))
# Having been created by twinx, ax3 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(ax3)
# Second, show the right spine.
ax3.spines["right"].set_visible(True)

ax3.plot(pnp_1d.grid/sc.nano, pnp_1d.charge_density, 
    label='Charge density', color='grey', linewidth=1, linestyle='-')

ax4.semilogy(
    pnp_1d.grid/sc.nano, 
    pnp_1d.concentration[0], marker='', color='tab:orange', 
    label='H3O+', linewidth=1, linestyle='-')
ax4.semilogy(x/sc.nano, np.ones(x.shape)*c[0], 
    label='bulk concentration', color='grey', linewidth=2, linestyle='-.')

ax4.semilogy(
    pnp_1d.grid/sc.nano, pnp_1d.concentration[1], 
    marker='', color='tab:blue', label='OH-', linewidth=1, linestyle='-')

ax1.set_xlabel('distance $x$ (nm)')
ax1.set_ylabel('potential $\phi$ (V)')
ax2.set_ylabel('concentration $c$ (mM)')
ax3.set_ylabel(r'charge density $\rho \> (\mathrm{C}\> \mathrm{m}^{-3})$')
ax4.set_ylabel('concentration $c$ (mM)')

ax1.legend(loc='upper left',  bbox_to_anchor=(1.3,1.02), fontsize=12, frameon=False)
ax2.legend(loc='center left', bbox_to_anchor=(1.3,0.5),  fontsize=12, frameon=False)
ax3.legend(loc='lower left',  bbox_to_anchor=(1.3,-0.02), fontsize=12, frameon=False)

fig.tight_layout()
plt.show()

# %% [markdown]
# # Sweep

# %%
pnp_list = []
for delta_u in u_list: 
    logger.info("")
    logger.info("### solution at u = %f V ###", delta_u)
    pnp_1d = PoissonNernstPlanckSystemFEniCSx(
        c=c, z=z, L=L, delta_u=delta_u, N=400)
        # solver="hybr", options={'xtol':1e-12})
    pnp_1d.use_standard_interface_bc()
    pnp_1d.solve();
    pnp_list.append((delta_u, pnp_1d))
    logger.info("")

# %%
#
# Copyright 2019-2020 Johannes Hoermann (U. Freiburg)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import os.path, re, sys
import numpy as np
from glob import glob
from cycler import cycler
from itertools import cycle
from itertools import groupby
import matplotlib.pyplot as plt


def right_align_legend(leg):
    hp = leg._legend_box.get_children()[1]
    for vp in hp.get_children():
        for row in vp.get_children():
            row.set_width(100)  # need to adapt this manually
            row.mode= "expand"
            row.align="right"

# sort file names as normal humans expect
# https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
def alpha_num_order(x):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    return [ convert(c) for c in re.split('([0-9]+)', x) ]


# dat_files = sorted(glob(glob_pattern),key=alpha_num_order)
N = len(pnp_list) # number of data sets
M = 2 # number of species

param_label = 'potential'

# matplotlib settings
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

# plt.rc('axes', prop_cycle=default_cycler)

plt.rc('font',   size=MEDIUM_SIZE)       # controls default text sizes
plt.rc('axes',   titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes',   labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick',  labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick',  labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titlex

plt.rcParams["figure.figsize"] = (16,10) # the standard figure size

plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 14
plt.rcParams["lines.markeredgewidth"]=1

# line styles
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
# linestyle_str = [
#     ('solid', 'solid'),      # Same as (0, ()) or '-'
#     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
#     ('dashed', 'dashed'),    # Same as '--'
#     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

# color maps for potential and concentration plots
cmap_u = plt.get_cmap('Reds')
cmap_c = [plt.get_cmap('Oranges'), plt.get_cmap('Blues')]

# general line style cycler
line_cycler =   cycler( linestyle = [ s for _,s in linestyle_tuple ] )

# potential anc concentration cyclers
u_cycler    =   cycler( color = cmap_u( np.linspace(0.4,0.8,N) ) )
u_cycler    =   len(line_cycler)*u_cycler + len(u_cycler)*line_cycler
c_cyclers   = [ cycler( color =   cmap( np.linspace(0.4,0.8,N) ) ) for cmap in cmap_c ]
c_cyclers   = [ len(line_cycler)*c_cycler + len(c_cycler)*line_cycler for c_cycler in c_cyclers ]

# https://matplotlib.org/3.1.1/tutorials/intermediate/constrainedlayout_guide.html
fig, (ax1,ax2,ax3) = plt.subplots(
    nrows=1, ncols=3, figsize=[24,7], constrained_layout=True)

ax1.set_xlabel('z (nm)')
ax1.set_ylabel('potential (V)')
ax2.set_xlabel('z (nm)')
ax2.set_ylabel('concentration (mM)')
ax3.set_xlabel('z (nm)')
ax3.set_ylabel('concentration (mM)')

# ax1.axvline(x=pnp.lambda_D()*1e9, label='Debye Length', color='grey', linestyle=':')
species_label = [
    '$[\mathrm{Na}^+], ' + param_label + '$',
    '$[\mathrm{Cl}^-], ' + param_label + '$']

# c_regex = re.compile(r'{}_(-?\d+(,\d+)*(\.\d+(e\d+)?)?)'.format(param))

c_graph_handles = [ [] for _ in range(M) ]
for (u_value, pnp), u_style, c_styles in zip(pnp_list,u_cycler,zip(*c_cyclers)):
    # print("Processing {:s}".format(f))
    # extract nominal concentration from file name
    #nominal_c = float( c_regex.search(f).group(1) )
    # nominal_c = 1e-4

    # dat = np.loadtxt(f,unpack=True)
    x = pnp.grid
    u = pnp.potential
    c = pnp.concentration

    # c_label = '{:> 4.1g}'.format(nominal_c)
    u_label = '{:> 4.1g} V'.format(u_value)
    # potential
    ax1.plot(x*1e9, u, marker=None, label=u_label, linewidth=1, **u_style)

    for i in range(c.shape[0]):
        # concentration
        ax2.plot(x*1e9, c[i], marker='',
            label=u_label, linewidth=2, **c_styles[i])
        # log-log concentration
        c_graph_handles[i].extend( ax3.loglog(x*1e9, c[i], marker='',
            label=u_label, linewidth=2, **c_styles[i]) )

# legend placement
# https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
u_legend = ax1.legend(loc='center right', title='potential, ${}$'.format(param_label), bbox_to_anchor=(-0.2,0.5))
first_c_legend  = ax3.legend(handles=c_graph_handles[0], title=species_label[0], loc='upper left', bbox_to_anchor=(1.00, 1.02) )
second_c_legend = ax3.legend(handles=c_graph_handles[1], title=species_label[1], loc='lower left', bbox_to_anchor=(1.00,-0.02) )
ax3.add_artist(first_c_legend) # add automatically removed first legend again
c_legends = [ first_c_legend, second_c_legend ]
legends = [ u_legend, *c_legends ]

for l in legends:
    right_align_legend(l)

# https://matplotlib.org/3.1.1/tutorials/intermediate/constrainedlayout_guide.html
for l in legends:
    l.set_in_layout(False)
# trigger a draw so that constrained_layout is executed once
# before we turn it off when printing....
fig.canvas.draw()
# we want the legend included in the bbox_inches='tight' calcs.
for l in legends:
    l.set_in_layout(True)
# we don't want the layout to change at this point.
fig.set_constrained_layout(False)

# fig.tight_layout(pad=3.0, w_pad=2.0, h_pad=1.0)
# plt.show()
# fig.savefig(figfile, bbox_inches='tight', dpi=100)

# %% [markdown]
# # Geometry & mesh on rough edge

# %% [markdown]
# Meshing adapted from https://docs.fenicsproject.org/dolfinx/v0.7.2/python/demos/demo_gmsh.html

# %%
debye_length

# %%
# units are in micro m = 1e-6 m, we convert to Debye lengths
x_rough_mum, y_rough_mum = np.loadtxt("data/profiles/Rough surface-3D-10x2-1-line (Parallel sliding direction).csv", skiprows=1, delimiter=',', usecols=(0,1), unpack=True, max_rows=1010)

# %%
x_rough_dimensionless = x_rough_mum*1e-6/debye_length

# %%
y_rough_dimensionless = y_rough_mum*1e-6/debye_length

# %%
# units are in micro m = 1e-6 m, we convert to Debye lengths
x_smooth_mum, y_smooth_mum = np.loadtxt("data/profiles/Smooth surface-3D-10x2-1-line (Parallel sliding direction).csv", skiprows=1, delimiter=',', usecols=(0,1), unpack=True, max_rows=1010)

# %%
x_smooth_dimensionless = x_smooth_mum*1e-6/debye_length

# %%
y_smooth_dimensionless = y_smooth_mum*1e-6/debye_length

# %%
# units are in micro m = 1e-6 m, we convert to Debye lengths
x_rough_vert_mum, y_rough_vert_mum = np.loadtxt("data/profiles/Rough surface-3D-10x2-1-line (Vertical sliding direction).csv", skiprows=1, delimiter=',', usecols=(0,1), unpack=True, max_rows=1007)

# %%
x_rough_vert_dimensionless = x_rough_vert_mum*1e-6/debye_length

# %%
y_rough_vert_dimensionless = y_rough_vert_mum*1e-6/debye_length

# %%
# units are in micro m = 1e-6 m, we convert to Debye lengths
x_smooth_vert_mum, y_smooth_vert_mum = np.loadtxt("data/profiles/Smooth surface-3D-10x2-1-line (Vertical sliding direction).csv", skiprows=1, delimiter=',', usecols=(0,1), unpack=True, max_rows=1007)

# %%
x_smooth_vert_dimensionless = x_smooth_vert_mum*1e-6/debye_length

# %%
y_smooth_vert_dimensionless = y_smooth_vert_mum*1e-6/debye_length

# %%
plt.plot(x_rough_dimensionless, y_rough_dimensionless, label="rough, parallel sliding direction")
plt.plot(x_smooth_dimensionless, y_smooth_dimensionless, label="smooth, prallel sliding direction")
plt.plot(x_rough_vert_dimensionless, y_rough_vert_dimensionless, label="rough, parallel sliding direction", linestyle='--')
plt.plot(x_smooth_vert_dimensionless, y_smooth_vert_dimensionless, label="smooth, vertical sliding direction", linestyle='--')
plt.ylabel("height y (Debye length $\lambda_D$)")
plt.xlabel("position x (Debye length $\lambda_D$)")
plt.legend()


# %%
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
    upper_boundary_mesh_size = 0.5

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

    # lines = [l1,l2,l3]
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

    # model.occ.synchronize()
    gmsh.model.geo.synchronize()
    
    # Define physical groups (optional)
    model.addPhysicalGroup(1, [l1], tag=1, name="Right boundary")
    model.addPhysicalGroup(1, [l2], tag=2, name="Upper boundary")
    model.addPhysicalGroup(1, [l3], tag=3, name="Left boundary")
    model.addPhysicalGroup(1, rough_edge, tag=4, name="Rough boundary")
    model.addPhysicalGroup(2, [surface], tag=5, name="Domain")

    # model.mesh.generate(dim=2)
    
    return model


# %% [markdown]
# ## Meshing rough edge geometries

# %%
gmsh.initialize()

# %%
gmsh.clear()

# %%
model = gmsh.model()

# %%
len(x_rough_dimensionless)

# %% [markdown]
# Geometries are created dimensionless. Height h given in Debye lengths.

# %%
model = gmsh_rectangle_with_single_rough_edge(
    model, "Rectangle with single rough edge", x_rough_dimensionless, y_rough_dimensionless, h=10,
    lower_boundary_mesh_size=0.1, upper_boundary_mesh_size=0.1)

# %%
gmsh.write('data/geometries/rectangle_with_single_rough_edge.geo_unrolled')

# %%
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.2)

# %%
gmsh.model.mesh.generate(2)

# %%
gmsh.write('data/meshes/rectangle_with_single_rough_edge.msh')

# %%
msh, ct, ft = gmshio.read_from_msh('data/meshes/rectangle_with_single_rough_edge_manual.msh', MPI.COMM_WORLD)

# %%
msh.name = "rectangle_with_single_rough_edge"
ct.name = f"{msh.name}_cells"
ft.name = f"{msh.name}_facets"

# %%
with XDMFFile(msh.comm, f"data/meshes/rectangle_with_single_rough_edge_mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w") as file:
    msh.topology.create_connectivity(1, 2)
    file.write_mesh(msh)
    file.write_meshtags(ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")
    file.write_meshtags(ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")

# %%
gmsh.initialize()

# %%
gmsh.clear()

# %%
model = gmsh.model()

# %%
len(x_rough_vert_dimensionless)

# %%
model = gmsh_rectangle_with_single_rough_edge(
    model, "Rectangle with single rough edge, vertical", x_rough_vert_dimensionless, y_rough_vert_dimensionless, h=10,
    lower_boundary_mesh_size=0.1, upper_boundary_mesh_size=0.1)

# %%
gmsh.write('data/geometries/rectangle_with_single_rough_edge_vertical.geo_unrolled')

# %%
msh, ct, ft = gmshio.read_from_msh('data/meshes/rectangle_with_single_rough_edge_vertical_manual.msh', MPI.COMM_WORLD)

# %%
msh.name = "rectangle_with_single_rough_edge_vertical"
ct.name = f"{msh.name}_cells"
ft.name = f"{msh.name}_facets"

# %%
with XDMFFile(msh.comm, f"data/meshes/rectangle_with_single_rough_edge_vertical_mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w") as file:
    msh.topology.create_connectivity(1, 2)
    file.write_mesh(msh)
    file.write_meshtags(ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")
    file.write_meshtags(ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")

# %% [markdown]
# ## Meshing smooth edge geometries

# %%
gmsh.initialize()

# %%
gmsh.clear()

# %%
model = gmsh.model()

# %%
model = gmsh_rectangle_with_single_rough_edge(model, "Rectangle with single smooth edge", 
                                              x_smooth_dimensionless, y_smooth_dimensionless, h=10,
                                              lower_boundary_mesh_size=0.1, upper_boundary_mesh_size=0.1)

# %%
gmsh.write('data/geometries/rectangle_with_single_smooth_edge.geo_unrolled')

# %%
msh, ct, ft = gmshio.read_from_msh('data/meshes/rectangle_with_single_smooth_edge_manual.msh', MPI.COMM_WORLD)

# %%
msh.name = "rectangle_with_single_smooth_edge"
ct.name = f"{msh.name}_cells"
ft.name = f"{msh.name}_facets"

# %%
with XDMFFile(msh.comm, f"data/meshes/rectangle_with_single_smooth_edge_mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w") as file:
    msh.topology.create_connectivity(1, 2)
    file.write_mesh(msh)
    file.write_meshtags(ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")
    file.write_meshtags(ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")

# %%
gmsh.initialize()

# %%
gmsh.clear()

# %%
model = gmsh.model()

# %%
model = gmsh_rectangle_with_single_rough_edge(model, "Rectangle with single smooth edge, vertical", 
                                              x_smooth_vert_dimensionless, y_smooth_vert_dimensionless, h=10,
                                              lower_boundary_mesh_size=0.1, upper_boundary_mesh_size=0.1)

# %%
gmsh.write('data/geometries/rectangle_with_single_smooth_edge_vertical.geo_unrolled')

# %%
msh, ct, ft = gmshio.read_from_msh('data/meshes/rectangle_with_single_smooth_edge_vertical_manual.msh', MPI.COMM_WORLD)

# %%
msh.name = "rectangle_with_single_smooth_edge_vertical"
ct.name = f"{msh.name}_cells"
ft.name = f"{msh.name}_facets"

# %%
with XDMFFile(msh.comm, f"data/meshes/rectangle_with_single_smooth_edge_vertical_mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w") as file:
    msh.topology.create_connectivity(1, 2)
    file.write_mesh(msh)
    file.write_meshtags(ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")
    file.write_meshtags(ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")

# %% [markdown]
# # 2d problem

# %%
c = [1.0e-4, 1.0e-4]  #mM
z = [1, -1]

# %%
delta_u = 0.05 # V

# %% [markdown]
# ## Solving the PDE

# %% [markdown]
# ### Rough edge, parallel

# %%
# define desired system
pnp_2d = PoissonNernstPlanckSystemFEniCSx2d(c=c, z=z, mesh_file="data/meshes/rectangle_with_single_rough_edge_manual.msh", scale_mesh=False)

# %%
pnp_2d.apply_concentration_dirichlet_bc(0, pnp_2d.c_scaled[0], 2)
pnp_2d.apply_concentration_dirichlet_bc(1, pnp_2d.c_scaled[1], 2)

# %%
pnp_2d.u_unit

# %%
delta_u_scaled = delta_u / pnp_2d.u_unit

# %%
delta_u_scaled

# %%
pnp_2d.apply_potential_dirichlet_bc(delta_u_scaled, 4)

# %%
pnp_2d.apply_potential_dirichlet_bc(0.0, 2)

# %%
pnp_2d.solve()

# %%
u_dimensionless, c_H3Op_dimensionless, c_OHm_dimensionless = pnp_2d.w.split()

# %%
adios4dolfinx.write_mesh(pnp_2d.mesh, 'data/checkpoint/rectangle_with_single_rough_edge_mesh.bp')

# %%
adios4dolfinx.write_function(u=pnp_2d.w, filename='data/checkpoint/rectangle_with_single_rough_edge_mesh.bp')

# %%
#H0 = ufl.FiniteElement("Discontinuous Lagrange", pnp_2d.mesh.ufl_cell(), 1)
H0 = basix.ufl.element("Lagrange", pnp_2d.mesh.basix_cell(), 1)

# %%
gdim = pnp_2d.mesh.geometry.dim

# %%
gdim

# %%
W0 = FunctionSpace(pnp_2d.mesh, H0)

# %%
u_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)

# %%
u_dimensionless_interpolated.interpolate(u_dimensionless)

# %%
c_H3Op_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)

# %%
c_H3Op_dimensionless_interpolated.interpolate(c_H3Op_dimensionless)

# %%
c_OHm_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)

# %%
c_OHm_dimensionless_interpolated.interpolate(c_OHm_dimensionless)

# %%
with XDMFFile(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_rough_edge_u_dimesionless.xdmf", "w") as file:
    file.write_mesh(pnp_2d.mesh)
    file.write_function(u_dimensionless_interpolated)

# %%
with XDMFFile(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_rough_edge_cH3Op_dimesionless.xdmf", "w") as file:
    file.write_mesh(pnp_2d.mesh)
    file.write_function(c_H3Op_dimensionless_interpolated)

# %%
with XDMFFile(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_rough_edge_cOHm_dimesionless.xdmf", "w") as file:
    file.write_mesh(pnp_2d.mesh)
    file.write_function(c_OHm_dimensionless_interpolated)

# %%
# the solution will be a function with X, Y, Z components corresponding to dimensionless u, cH3Op, and cOH-
# with VTXWriter(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_rough_edge_solution_dimensionless.bp", pnp_2d.w, "bp4") as f:
#    f.write(0.0)

# %% [markdown]
# ### Rough edge, vertical

# %%
# define desired system
pnp_2d = PoissonNernstPlanckSystemFEniCSx2d(c=c, z=z, mesh_file="data/meshes/rectangle_with_single_rough_edge_vertical_manual.msh", scale_mesh=False)

# %%
pnp_2d.apply_concentration_dirichlet_bc(0, pnp_2d.c_scaled[0], 2)
pnp_2d.apply_concentration_dirichlet_bc(1, pnp_2d.c_scaled[1], 2)

# %%
pnp_2d.u_unit

# %%
delta_u_scaled = delta_u / pnp_2d.u_unit

# %%
delta_u_scaled

# %%
pnp_2d.apply_potential_dirichlet_bc(delta_u_scaled, 4)

# %%
pnp_2d.apply_potential_dirichlet_bc(0.0, 2)

# %%
pnp_2d.solve()

# %%
u_dimensionless, c_H3Op_dimensionless, c_OHm_dimensionless = pnp_2d.w.split()

# %%
adios4dolfinx.write_mesh(pnp_2d.mesh, 'data/checkpoint/rectangle_with_single_rough_edge_vertical_mesh.bp')

# %%
adios4dolfinx.write_function(u=pnp_2d.w, filename='data/checkpoint/rectangle_with_single_rough_edge_vertical_mesh.bp')

# %%
#H0 = ufl.FiniteElement("Discontinuous Lagrange", pnp_2d.mesh.ufl_cell(), 1)
H0 = basix.ufl.element("Lagrange", pnp_2d.mesh.basix_cell(), 1)

# %%
gdim = pnp_2d.mesh.geometry.dim

# %%
gdim

# %%
W0 = FunctionSpace(pnp_2d.mesh, H0)

# %%
u_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)

# %%
u_dimensionless_interpolated.interpolate(u_dimensionless)

# %%
c_H3Op_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)

# %%
c_H3Op_dimensionless_interpolated.interpolate(c_H3Op_dimensionless)

# %%
c_OHm_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)

# %%
c_OHm_dimensionless_interpolated.interpolate(c_OHm_dimensionless)

# %%
with XDMFFile(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_rough_edge_vertical_u_dimesionless.xdmf", "w") as file:
    file.write_mesh(pnp_2d.mesh)
    file.write_function(u_dimensionless_interpolated)

# %%
with XDMFFile(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_rough_edge_vertical_cH3Op_dimesionless.xdmf", "w") as file:
    file.write_mesh(pnp_2d.mesh)
    file.write_function(c_H3Op_dimensionless_interpolated)

# %%
with XDMFFile(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_rough_edge_vertical_cOHm_dimesionless.xdmf", "w") as file:
    file.write_mesh(pnp_2d.mesh)
    file.write_function(c_OHm_dimensionless_interpolated)

# %%
# the solution will be a function with X, Y, Z components corresponding to dimensionless u, cH3Op, and cOH-
# with VTXWriter(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_rough_edge_vertical_solution_dimensionless.bp", pnp_2d.w, "bp4") as f:
#    f.write(0.0)

# %% [markdown]
# ### Smooth edge, parallel

# %%
# define desired system
pnp_2d = PoissonNernstPlanckSystemFEniCSx2d(c=c, z=z, mesh_file="data/meshes/rectangle_with_single_smooth_edge_manual.msh", scale_mesh=False)

# %%
pnp_2d.apply_concentration_dirichlet_bc(0, pnp_2d.c_scaled[0], 2)
pnp_2d.apply_concentration_dirichlet_bc(1, pnp_2d.c_scaled[1], 2)

# %%
pnp_2d.u_unit

# %%
delta_u_scaled = delta_u / pnp_2d.u_unit

# %%
delta_u_scaled

# %%
pnp_2d.apply_potential_dirichlet_bc(delta_u_scaled, 4)

# %%
pnp_2d.apply_potential_dirichlet_bc(0.0, 2)

# %%
pnp_2d.solve()

# %%
u_dimensionless, c_H3Op_dimensionless, c_OHm_dimensionless = pnp_2d.w.split()

# %%
adios4dolfinx.write_mesh(pnp_2d.mesh, 'data/checkpoint/rectangle_with_single_smooth_edge_mesh.bp')

# %%
adios4dolfinx.write_function(u=pnp_2d.w, filename='data/checkpoint/rectangle_with_single_smooth_edge_mesh.bp')

# %%
#H0 = ufl.FiniteElement("Discontinuous Lagrange", pnp_2d.mesh.ufl_cell(), 1)
H0 = basix.ufl.element("Lagrange", pnp_2d.mesh.basix_cell(), 1)

# %%
gdim = pnp_2d.mesh.geometry.dim

# %%
gdim

# %%
W0 = FunctionSpace(pnp_2d.mesh, H0)

# %%
u_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)

# %%
u_dimensionless_interpolated.interpolate(u_dimensionless)

# %%
c_H3Op_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)

# %%
c_H3Op_dimensionless_interpolated.interpolate(c_H3Op_dimensionless)

# %%
c_OHm_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)

# %%
c_OHm_dimensionless_interpolated.interpolate(c_OHm_dimensionless)

# %%
with XDMFFile(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_smooth_edge_u_dimesionless.xdmf", "w") as file:
    file.write_mesh(pnp_2d.mesh)
    file.write_function(u_dimensionless_interpolated)

# %%
with XDMFFile(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_smooth_edge_cH3Op_dimesionless.xdmf", "w") as file:
    file.write_mesh(pnp_2d.mesh)
    file.write_function(c_H3Op_dimensionless_interpolated)

# %%
with XDMFFile(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_smooth_edge_cOHm_dimesionless.xdmf", "w") as file:
    file.write_mesh(pnp_2d.mesh)
    file.write_function(c_OHm_dimensionless_interpolated)

# %%
# the solution will be a function with X, Y, Z components corresponding to dimensionless u, cH3Op, and cOH-
# with VTXWriter(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_smooth_edge_solution_dimensionless.bp", pnp_2d.w, "bp4") as f:
#    f.write(0.0)

# %% [markdown]
# ### Smooth edge, vertical

# %%
# define desired system
pnp_2d = PoissonNernstPlanckSystemFEniCSx2d(c=c, z=z, mesh_file="data/meshes/rectangle_with_single_smooth_edge_vertical_manual.msh", scale_mesh=False)

# %%
pnp_2d.apply_concentration_dirichlet_bc(0, pnp_2d.c_scaled[0], 2)
pnp_2d.apply_concentration_dirichlet_bc(1, pnp_2d.c_scaled[1], 2)

# %%
pnp_2d.u_unit

# %%
delta_u_scaled = delta_u / pnp_2d.u_unit

# %%
delta_u_scaled

# %%
pnp_2d.apply_potential_dirichlet_bc(delta_u_scaled, 4)

# %%
pnp_2d.apply_potential_dirichlet_bc(0.0, 2)

# %%
pnp_2d.solve()

# %%
u_dimensionless, c_H3Op_dimensionless, c_OHm_dimensionless = pnp_2d.w.split()

# %%
adios4dolfinx.write_mesh(pnp_2d.mesh, 'data/checkpoint/rectangle_with_single_smooth_edge_vertical_mesh.bp')

# %%
adios4dolfinx.write_function(u=pnp_2d.w, filename='data/checkpoint/rectangle_with_single_smooth_edge_vertical_mesh.bp')

# %%
#H0 = ufl.FiniteElement("Discontinuous Lagrange", pnp_2d.mesh.ufl_cell(), 1)
H0 = basix.ufl.element("Lagrange", pnp_2d.mesh.basix_cell(), 1)

# %%
gdim = pnp_2d.mesh.geometry.dim

# %%
gdim

# %%
W0 = FunctionSpace(pnp_2d.mesh, H0)

# %%
u_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)

# %%
u_dimensionless_interpolated.interpolate(u_dimensionless)

# %%
c_H3Op_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)

# %%
c_H3Op_dimensionless_interpolated.interpolate(c_H3Op_dimensionless)

# %%
c_OHm_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)

# %%
c_OHm_dimensionless_interpolated.interpolate(c_OHm_dimensionless)

# %%
with XDMFFile(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_smooth_edge_vertical_u_dimesionless.xdmf", "w") as file:
    file.write_mesh(pnp_2d.mesh)
    file.write_function(u_dimensionless_interpolated)

# %%
with XDMFFile(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_smooth_edge_vertical_cH3Op_dimesionless.xdmf", "w") as file:
    file.write_mesh(pnp_2d.mesh)
    file.write_function(c_H3Op_dimensionless_interpolated)

# %%
with XDMFFile(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_smooth_edge_vertical_cOHm_dimesionless.xdmf", "w") as file:
    file.write_mesh(pnp_2d.mesh)
    file.write_function(c_OHm_dimensionless_interpolated)

# %%
# the solution will be a function with X, Y, Z components corresponding to dimensionless u, cH3Op, and cOH-
# with VTXWriter(pnp_2d.mesh.comm, "data/solutions/rectangle_with_single_smooth_edge_vertical_solution_dimensionless.bp", pnp_2d.w, "bp4") as f:
#    f.write(0.0)

# %% [markdown]
# # Analysis of solutionf for rough profile, parallel

# %% [markdown]
# ## Read solution from file

# %%
checkpoint_filename = 'data/checkpoint/rectangle_with_single_rough_edge_mesh.bp'

# %%
mesh = adios4dolfinx.read_mesh(MPI.COMM_WORLD, checkpoint_filename, "BP4", dolfinx.mesh.GhostMode.none)

# %%
P = basix.ufl.element('Lagrange', mesh.basix_cell(), 3)
elements = [P] * 3
H = basix.ufl.mixed_element(elements)
W = dolfinx.fem.FunctionSpace(mesh, H)

# %%
w = dolfinx.fem.Function(W)

# %%
w.function_space.mesh

# %%
adios4dolfinx.read_function(w, checkpoint_filename)

# %%
w

# %%
w.x.array.shape

# %%
u_function, c_H3Op_function, c_OHm_function = w.split()

# %%
u_function.x.array.shape

# %%
gdim = mesh.geometry.dim

# %%
gdim

# %%
H0 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
W0 = FunctionSpace(mesh, H0)

# %%
u_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)
u_dimensionless_interpolated.interpolate(u_function)

# %%
u_dimensionless_interpolated.x.array.shape

# %%
c_H3Op_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)
c_H3Op_dimensionless_interpolated.interpolate(c_H3Op_function)

# %%
c_OHm_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)
c_OHm_dimensionless_interpolated.interpolate(c_OHm_function)

# %% [markdown]
# ## Plot with pyvista

# %%
pyvista.start_xvfb()

# %%
topology, cell_types, geometry = plot.vtk_mesh(mesh)

# %%
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# %%
grid.point_data["u"] = u_dimensionless_interpolated.x.array.real
grid.set_active_scalars("u")

# %%
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

# %%
plotter.bounds

# %%
# plotter.set_position([0,0,-1])

# %%
plotter.bounds

# %%
# plotter.set_scale(10,10,10)

# %%
# plotter.view_xy()

# %%
# plotter.camera_position = "xz"
# plotter.camera.azimuth = 45
# plotter.camera.elevation = 30
plotter.camera.zoom(20)

# %%
plotter.show()

# %%
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# %%
grid.point_data["c_H3Op"] = c_H3Op_dimensionless_interpolated.x.array.real
grid.set_active_scalars("c_H3Op")

# %%
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

# %%
plotter.camera.zoom(20)

# %%
plotter.show()

# %%
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# %%
grid.point_data["c_OHm"] = c_OHm_dimensionless_interpolated.x.array.real
grid.set_active_scalars("c_OHm")

# %%
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

# %%
plotter.camera.zoom(20)

# %%
plotter.show()

# %% [markdown]
# ## Surface integrals

# %% [markdown]
# ### Surface excess in 1d case

# %%
c

# %%
z

# %%
delta_u

# %%
debye_length

# %%
L = 10*debye_length

# %%
L

# %%
pnp_1d = PoissonNernstPlanckSystemFEniCSx(c=c, z=z, delta_u=delta_u, L=L, N=1000)

# %%
pnp_1d.use_standard_interface_bc()

# %%
potential_ref_unitless, c_ref_unitless, _ = pnp_1d.solve()

# %%
c_H3Op_ref_unitless = c_ref_unitless[0,:]
c_OHm_ref_unitless = c_ref_unitless[1,:]

# %%
x_ref_unitless = pnp_1d.grid_dimensionless

# %%
plt.plot(x_ref_unitless, c_H3Op_ref_unitless)
plt.plot(x_ref_unitless, c_OHm_ref_unitless)

# %%
plt.plot(x_ref_unitless, c_H3Op_ref_unitless-1)
plt.plot(x_ref_unitless, c_OHm_ref_unitless-1)

# %%
N_excess_H3Op_ref = np.trapz(c_H3Op_ref_unitless-1., x_ref_unitless)

# %%
N_excess_OHm_ref = np.trapz(c_OHm_ref_unitless-1., x_ref_unitless)

# %%
N_excess_H3Op_ref

# %%
N_excess_OHm_ref

# %%
dx = np.mean(x_rough_dimensionless[1:]-x_rough_dimensionless[:-1])

# %%
dx

# %%
x_rough_dimensionless[0]

# %%
A_apparent_rough = x_rough_dimensionless[-1] - x_rough_dimensionless[0] + dx

# %%
A_apparent_rough

# %%
N_excess_OHm_ref*A_apparent_rough

# %%
N_excess_H3Op_ref*A_apparent_rough

# %% [markdown]
# ### Reference amount of species

# %%
reference_concentration_dimensionless = dolfinx.fem.Constant(mesh, default_scalar_type(1))

# %%
N_unit_concentration_expression = dolfinx.fem.form(reference_concentration_dimensionless*ufl.dx)

# %%
N_unit_concentration = dolfinx.fem.assemble_scalar(N_unit_concentration_expression)

# %%
N_unit_concentration

# %% [markdown]
# ### Amount of H3Op

# %%
N_H3Op_dimensionless_expression = dolfinx.fem.form(c_H3Op_dimensionless_interpolated * ufl.dx)

# %%
N_H3Op_dimensionless_local = dolfinx.fem.assemble_scalar(N_H3Op_dimensionless_expression)

# %%
N_H3Op_dimensionless_local

# %%
N_H3Op_dimensionless = mesh.comm.allreduce(N_H3Op_dimensionless_local, op=MPI.SUM)

# %%
N_H3Op_dimensionless

# %%
N_H3Op_excess = N_H3Op_dimensionless - N_unit_concentration

# %%
N_H3Op_excess

# %%
N_H3Op_excess / N_unit_concentration  # depletion per volume

# %%
dN_H3Op_rough = N_H3Op_excess - N_excess_H3Op_ref*A_apparent_rough

# %%
dN_H3Op_rough

# %% [markdown]
# ### Amount of OHm

# %%
N_OHm_dimensionless_expression = dolfinx.fem.form(c_OHm_dimensionless_interpolated * ufl.dx)

# %%
N_OHm_dimensionless_local = dolfinx.fem.assemble_scalar(N_OHm_dimensionless_expression)

# %%
N_OHm_dimensionless_local

# %%
N_OHm_dimensionless = mesh.comm.allreduce(N_OHm_dimensionless_local, op=MPI.SUM)

# %%
N_OHm_dimensionless

# %%
N_OHm_excess = N_OHm_dimensionless - N_unit_concentration

# %%
N_OHm_excess

# %%
N_OHm_excess / N_unit_concentration  # excess per volume

# %%
# excess compared to flat surface
dN_OHm_rough = N_OHm_excess - N_excess_OHm_ref*A_apparent_rough

# %%
dN_OHm_rough

# %% [markdown]
# ## Line integrals

# %%
dx = np.mean(x_rough_dimensionless[1:]-x_rough_dimensionless[:-1])


y_mean = np.mean(y_rough_dimensionless)
y_zero_aligned = y_rough_dimensionless - y_mean
x_zero_aligned = x_rough_dimensionless - x_rough_dimensionless[0] + dx

x0 = 0
x1 = x_zero_aligned[-1] + dx

# %%
y_mean

# %%
A_apparent_rough

# %%
tol = 0.0001
y_max = 10
N_points = 1001

c_ref = 1 # dimensionless

cH3Op_integrals = []
cOHm_integrals = []

dN_H3Op_integrals = []
dN_OHm_integrals = []

bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

for x, y_min in zip(x_zero_aligned, y_zero_aligned):
    y_grid = np.linspace(y_min + tol, y_max - tol, N_points)
    # dy = np.mean(y_grid[1:]-y_grid[:-1])

    points = np.zeros((3, N_points))
    points[0,:] = x
    points[1,:] = y_grid
    
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    cH3Op_values = c_H3Op_dimensionless_interpolated.eval(points_on_proc, cells).T
    cOHm_values = c_OHm_dimensionless_interpolated.eval(points_on_proc, cells).T

    # line integrals
    cH3Op_integral = np.trapz(y=cH3Op_values, x=y_grid)
    cOHm_integral = np.trapz(y=cOHm_values, x=y_grid)

    dN_cH3Op_excess_integral = np.trapz(y=cH3Op_values-c_ref, x=y_grid) - N_excess_H3Op_ref
    dN_OHm_excess_integral = np.trapz(y=cOHm_values-c_ref, x=y_grid) - N_excess_OHm_ref

    cH3Op_integrals.append(cH3Op_integral)
    cOHm_integrals.append(cOHm_integral)

    dN_H3Op_integrals.append(dN_cH3Op_excess_integral)
    dN_OHm_integrals.append(dN_OHm_excess_integral)

# %%
x_rough_dimensionless.shape

# %%
y_rough_dimensionless.shape

# %%
cH3Op_integrals = np.array(cH3Op_integrals).reshape(x_rough_dimensionless.shape[0])

# %%
cOHm_integrals = np.array(cOHm_integrals).reshape(x_rough_dimensionless.shape[0])

# %%
dN_H3Op_integrals = np.array(dN_H3Op_integrals).reshape(x_rough_dimensionless.shape[0])

# %%
dN_OHm_integrals = np.array(dN_OHm_integrals).reshape(x_rough_dimensionless.shape[0])

# %%
out_array = np.vstack([
    x_rough_dimensionless, y_rough_dimensionless, cH3Op_integrals, cOHm_integrals, dN_H3Op_integrals, dN_OHm_integrals]).T

# %%
np.savetxt('data/integrals/rough_parallel_line_integrals.csv', 
           out_array,
           delimiter=',',
           header='x [Debye lengths], y [Debye lengths], dimensionless cH3Op integrated along y-direction, dimensionless cOHM integrated along y-direction, excess amount of H3Op compared to flat surface, excess amount of H3Op compared to flat surface')

# %% [markdown]
# ### cH3Op integrals

# %%
plt.plot(x_rough_dimensionless, cH3Op_integrals)

# %% [markdown]
# ### cOHm integrals

# %%
plt.plot(x_rough_dimensionless, cOHm_integrals)

# %% [markdown]
# ### H3Op excess 

# %%
plt.plot(x_rough_dimensionless, y_rough_dimensionless, label="profile")
plt.plot(x_rough_dimensionless, dN_H3Op_integrals, linestyle=':', label="dN H3Op")
plt.legend()

# %% [markdown]
# ### OHm excess

# %%
plt.plot(x_rough_dimensionless, y_rough_dimensionless, label="profile")
plt.plot(x_rough_dimensionless, dN_OHm_integrals, linestyle=':', label="dN OHm")
plt.legend()

# %% [markdown]
# ### Zooms

# %%
x_rough_dimensionless.shape

# %%
subset = slice(100,200)

# %%
plt.plot(x_rough_dimensionless[subset], y_rough_dimensionless[subset], label="profile")
plt.plot(x_rough_dimensionless[subset], dN_OHm_integrals[subset], linestyle=':', label="dN OHm")
plt.legend()

# %%
plt.plot(x_rough_dimensionless[subset], y_rough_dimensionless[subset], label="profile")
plt.plot(x_rough_dimensionless[subset], dN_H3Op_integrals[subset], linestyle=':', label="dN H3Op")
plt.legend()

# %%
subset = slice(300,400)

# %%
plt.plot(x_rough_dimensionless[subset], y_rough_dimensionless[subset], label="profile")
plt.plot(x_rough_dimensionless[subset], dN_OHm_integrals[subset], linestyle=':', label="dN OHm")
plt.legend()

# %%
plt.plot(x_rough_dimensionless[subset], y_rough_dimensionless[subset], label="profile")
plt.plot(x_rough_dimensionless[subset], dN_H3Op_integrals[subset], linestyle=':', label="dN H3Op")
plt.hlines(y=0,xmin=x_rough_dimensionless[subset][0], xmax=x_rough_dimensionless[subset][-1], linestyle='--', color='gray')

# %%
plt.plot(x_rough_dimensionless[subset], y_rough_dimensionless[subset], label="profile")
plt.plot(x_rough_dimensionless[subset], dN_H3Op_integrals[subset], linestyle=':', label="dN H3Op")
plt.plot(x_rough_dimensionless[subset], dN_OHm_integrals[subset], linestyle=':', label="dN OHm")
plt.hlines(y=0,xmin=x_rough_dimensionless[subset][0], xmax=x_rough_dimensionless[subset][-1], linestyle='--', color='gray')
plt.legend()

# %% [markdown]
# # Analysis of solutionf for rough profile, vertical

# %% [markdown]
# ## Read solution from file

# %%
checkpoint_filename = 'data/checkpoint/rectangle_with_single_rough_edge_vertical_mesh.bp'

# %%
mesh = adios4dolfinx.read_mesh(MPI.COMM_WORLD, checkpoint_filename, "BP4", dolfinx.mesh.GhostMode.none)

# %%
P = basix.ufl.element('Lagrange', mesh.basix_cell(), 3)
elements = [P] * 3
H = basix.ufl.mixed_element(elements)
W = dolfinx.fem.FunctionSpace(mesh, H)

# %%
w = dolfinx.fem.Function(W)

# %%
w.function_space.mesh

# %%
adios4dolfinx.read_function(w, checkpoint_filename)

# %%
w

# %%
u_function, c_H3Op_function, c_OHm_function = w.split()

# %%
gdim = mesh.geometry.dim

# %%
gdim

# %%
H0 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
W0 = FunctionSpace(mesh, H0)

# %%
u_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)
u_dimensionless_interpolated.interpolate(u_function)

# %%
c_H3Op_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)
c_H3Op_dimensionless_interpolated.interpolate(c_H3Op_function)

# %%
c_OHm_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)
c_OHm_dimensionless_interpolated.interpolate(c_OHm_function)

# %% [markdown]
# ## Plot with pyvista

# %%
pyvista.start_xvfb()

# %%
topology, cell_types, geometry = plot.vtk_mesh(mesh)

# %%
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# %%
grid.point_data["u"] = u_dimensionless_interpolated.x.array.real
grid.set_active_scalars("u")

# %%
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

# %%
plotter.camera.zoom(20)

# %%
plotter.show()

# %%
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# %%
grid.point_data["c_H3Op"] = c_H3Op_dimensionless_interpolated.x.array.real
grid.set_active_scalars("c_H3Op")

# %%
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

# %%
plotter.camera.zoom(20)

# %%
plotter.show()

# %%
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# %%
grid.point_data["c_OHm"] = c_OHm_dimensionless_interpolated.x.array.real
grid.set_active_scalars("c_OHm")

# %%
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

# %%
plotter.camera.zoom(20)

# %%
plotter.show()

# %% [markdown]
# ## Surface integrals

# %% [markdown]
# ### Surface excess in 1d case

# %%
c

# %%
z

# %%
delta_u

# %%
debye_length

# %%
L = 10*debye_length

# %%
L

# %%
pnp_1d = PoissonNernstPlanckSystemFEniCSx(c=c, z=z, delta_u=delta_u, L=L, N=1000)

# %%
pnp_1d.use_standard_interface_bc()

# %%
potential_ref_unitless, c_ref_unitless, _ = pnp_1d.solve()

# %%
c_H3Op_ref_unitless = c_ref_unitless[0,:]
c_OHm_ref_unitless = c_ref_unitless[1,:]

# %%
x_ref_unitless = pnp_1d.grid_dimensionless

# %%
plt.plot(x_ref_unitless, c_H3Op_ref_unitless)
plt.plot(x_ref_unitless, c_OHm_ref_unitless)

# %%
plt.plot(x_ref_unitless, c_H3Op_ref_unitless-1)
plt.plot(x_ref_unitless, c_OHm_ref_unitless-1)

# %%
c_H3Op_ref_unitless.shape

# %%
N_excess_H3Op_ref = np.trapz(c_H3Op_ref_unitless-1., x_ref_unitless)

# %%
N_excess_OHm_ref = np.trapz(c_OHm_ref_unitless-1., x_ref_unitless)

# %%
N_excess_H3Op_ref

# %%
N_excess_OHm_ref

# %%
dx = np.mean(x_rough_vert_dimensionless[1:]-x_rough_vert_dimensionless[:-1])

# %%
dx

# %%
x_rough_dimensionless[0]

# %%
A_apparent_rough_vertical = x_rough_vert_dimensionless[-1] - x_rough_vert_dimensionless[0] + dx

# %%
A_apparent_rough_vertical

# %%
N_excess_OHm_ref*A_apparent_rough_vertical

# %%
N_excess_H3Op_ref*A_apparent_rough_vertical

# %% [markdown]
# ### Reference amount of species

# %%
reference_concentration_dimensionless = dolfinx.fem.Constant(mesh, default_scalar_type(1))

# %%
N_unit_concentration_expression = dolfinx.fem.form(reference_concentration_dimensionless*ufl.dx)

# %%
N_unit_concentration = dolfinx.fem.assemble_scalar(N_unit_concentration_expression)

# %%
N_unit_concentration

# %% [markdown]
# ### Amount of H3Op

# %%
N_H3Op_dimensionless_expression = dolfinx.fem.form(c_H3Op_dimensionless_interpolated * ufl.dx)

# %%
N_H3Op_dimensionless_local = dolfinx.fem.assemble_scalar(N_H3Op_dimensionless_expression)

# %%
N_H3Op_dimensionless_local

# %%
N_H3Op_dimensionless = mesh.comm.allreduce(N_H3Op_dimensionless_local, op=MPI.SUM)

# %%
N_H3Op_dimensionless

# %%
N_H3Op_excess = N_H3Op_dimensionless - N_unit_concentration

# %%
N_H3Op_excess

# %%
N_H3Op_excess / N_unit_concentration  # depletion per volume

# %%
dN_H3Op_rough_vertical = N_H3Op_excess - N_excess_H3Op_ref*A_apparent_rough_vertical

# %%
dN_H3Op_rough_vertical

# %% [markdown]
# ### Amount of OHm

# %%
N_OHm_dimensionless_expression = dolfinx.fem.form(c_OHm_dimensionless_interpolated * ufl.dx)

# %%
N_OHm_dimensionless_local = dolfinx.fem.assemble_scalar(N_OHm_dimensionless_expression)

# %%
N_OHm_dimensionless_local

# %%
N_OHm_dimensionless = mesh.comm.allreduce(N_OHm_dimensionless_local, op=MPI.SUM)

# %%
N_OHm_dimensionless

# %%
N_OHm_excess = N_OHm_dimensionless - N_unit_concentration

# %%
N_OHm_excess

# %%
N_OHm_excess / N_unit_concentration  # excess per volume

# %%
# excess compared to flat surface
dN_OHm_rough_vertical = N_OHm_excess - N_excess_OHm_ref*A_apparent_rough_vertical

# %%
dN_OHm_rough_vertical

# %% [markdown]
# ## Line integrals

# %%
dx = np.mean(x_rough_vert_dimensionless[1:]-x_rough_vert_dimensionless[:-1])


y_mean = np.mean(y_rough_vert_dimensionless)
y_zero_aligned = y_rough_vert_dimensionless - y_mean
x_zero_aligned = x_rough_vert_dimensionless - x_rough_vert_dimensionless[0] + dx

x0 = 0
x1 = x_zero_aligned[-1] + dx

# %%
y_mean

# %%
A_apparent_rough_vertical

# %%
tol = 0.0001
y_max = 10
N_points = 1001

c_ref = 1 # dimensionless

cH3Op_integrals = []
cOHm_integrals = []

dN_H3Op_integrals = []
dN_OHm_integrals = []

bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

for x, y_min in zip(x_zero_aligned, y_zero_aligned):
    y_grid = np.linspace(y_min + tol, y_max - tol, N_points)
    # dy = np.mean(y_grid[1:]-y_grid[:-1])

    points = np.zeros((3, N_points))
    points[0,:] = x
    points[1,:] = y_grid
    
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    cH3Op_values = c_H3Op_dimensionless_interpolated.eval(points_on_proc, cells).T
    cOHm_values = c_OHm_dimensionless_interpolated.eval(points_on_proc, cells).T

    # line integrals
    cH3Op_integral = np.trapz(y=cH3Op_values, x=y_grid)
    cOHm_integral = np.trapz(y=cOHm_values, x=y_grid)

    dN_cH3Op_excess_integral = np.trapz(y=cH3Op_values-c_ref, x=y_grid) - N_excess_H3Op_ref
    dN_OHm_excess_integral = np.trapz(y=cOHm_values-c_ref, x=y_grid) - N_excess_OHm_ref

    cH3Op_integrals.append(cH3Op_integral)
    cOHm_integrals.append(cOHm_integral)

    dN_H3Op_integrals.append(dN_cH3Op_excess_integral)
    dN_OHm_integrals.append(dN_OHm_excess_integral)

# %%
cH3Op_integrals = np.array(cH3Op_integrals).reshape(x_rough_vert_dimensionless.shape[0])

# %%
cOHm_integrals = np.array(cOHm_integrals).reshape(x_rough_vert_dimensionless.shape[0])

# %%
dN_H3Op_integrals = np.array(dN_H3Op_integrals).reshape(x_rough_vert_dimensionless.shape[0])

# %%
dN_OHm_integrals = np.array(dN_OHm_integrals).reshape(x_rough_vert_dimensionless.shape[0])

# %%
out_array = np.vstack([
    x_rough_vert_dimensionless, y_rough_vert_dimensionless, cH3Op_integrals, cOHm_integrals, dN_H3Op_integrals, dN_OHm_integrals]).T

# %%
np.savetxt('data/integrals/rough_vertical_line_integrals.csv', 
           out_array,
           delimiter=',',
           header='x [Debye lengths], y [Debye lengths], dimensionless cH3Op integrated along y-direction, dimensionless cOHM integrated along y-direction, excess amount of H3Op compared to flat surface, excess amount of H3Op compared to flat surface')

# %% [markdown]
# ### cH3Op integrals

# %%
plt.plot(x_rough_vert_dimensionless, cH3Op_integrals)

# %% [markdown]
# ### cOHm integrals

# %%
plt.plot(x_rough_vert_dimensionless, cOHm_integrals)

# %% [markdown]
# ### H3Op excess 

# %%
plt.plot(x_rough_vert_dimensionless, y_rough_vert_dimensionless, label="profile")
plt.plot(x_rough_vert_dimensionless, dN_H3Op_integrals, linestyle=':', label="dN H3Op")
plt.legend()

# %% [markdown]
# ### OHm excess

# %%
plt.plot(x_rough_vert_dimensionless, y_rough_vert_dimensionless, label="profile")
plt.plot(x_rough_vert_dimensionless, dN_OHm_integrals, linestyle=':', label="dN OHm")
plt.legend()

# %% [markdown]
# ### Zooms

# %%
x_rough_dimensionless.shape

# %%
subset = slice(100,200)

# %%
plt.plot(x_rough_vert_dimensionless[subset], y_rough_vert_dimensionless[subset], label="profile")
plt.plot(x_rough_vert_dimensionless[subset], dN_OHm_integrals[subset], linestyle=':', label="dN OHm")
plt.legend()

# %%
plt.plot(x_rough_vert_dimensionless[subset], y_rough_vert_dimensionless[subset], label="profile")
plt.plot(x_rough_vert_dimensionless[subset], dN_H3Op_integrals[subset], linestyle=':', label="dN H3Op")
plt.legend()

# %%
subset = slice(600,800)

# %%
plt.plot(x_rough_vert_dimensionless[subset], y_rough_vert_dimensionless[subset], label="profile")
plt.plot(x_rough_vert_dimensionless[subset], dN_OHm_integrals[subset], linestyle=':', label="dN OHm")
plt.legend()

# %%
plt.plot(x_rough_vert_dimensionless[subset], y_rough_vert_dimensionless[subset], label="profile")
plt.plot(x_rough_vert_dimensionless[subset], dN_H3Op_integrals[subset], linestyle=':', label="dN H3Op")
plt.hlines(y=0,xmin=x_rough_vert_dimensionless[subset][0], xmax=x_rough_vert_dimensionless[subset][-1], linestyle='--', color='gray')

# %%
plt.plot(x_rough_vert_dimensionless[subset], y_rough_vert_dimensionless[subset], label="profile")
plt.plot(x_rough_vert_dimensionless[subset], dN_H3Op_integrals[subset], linestyle=':', label="dN H3Op")
plt.plot(x_rough_vert_dimensionless[subset], dN_OHm_integrals[subset], linestyle=':', label="dN OHm")
plt.hlines(y=0,xmin=x_rough_vert_dimensionless[subset][0], xmax=x_rough_vert_dimensionless[subset][-1], linestyle='--', color='gray')
plt.legend()

# %% [markdown]
# # Analysis of solutionf for smooth profile, parallel

# %% [markdown]
# ## Read solution from file

# %%
checkpoint_filename = 'data/checkpoint/rectangle_with_single_smooth_edge_mesh.bp'

# %%
mesh = adios4dolfinx.read_mesh(MPI.COMM_WORLD, checkpoint_filename, "BP4", dolfinx.mesh.GhostMode.none)

# %%
P = basix.ufl.element('Lagrange', mesh.basix_cell(), 3)
elements = [P] * 3
H = basix.ufl.mixed_element(elements)
W = dolfinx.fem.FunctionSpace(mesh, H)

# %%
w = dolfinx.fem.Function(W)

# %%
w.function_space.mesh

# %%
adios4dolfinx.read_function(w, checkpoint_filename)

# %%
w

# %%
w.x.array.shape

# %%
u_function, c_H3Op_function, c_OHm_function = w.split()

# %%
u_function.x.array.shape

# %%
gdim = mesh.geometry.dim

# %%
gdim

# %%
H0 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
W0 = FunctionSpace(mesh, H0)

# %%
u_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)
u_dimensionless_interpolated.interpolate(u_function)

# %%
u_dimensionless_interpolated.x.array.shape

# %%
c_H3Op_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)
c_H3Op_dimensionless_interpolated.interpolate(c_H3Op_function)

# %%
c_OHm_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)
c_OHm_dimensionless_interpolated.interpolate(c_OHm_function)

# %% [markdown]
# ## Plot with pyvista

# %%
pyvista.start_xvfb()

# %%
topology, cell_types, geometry = plot.vtk_mesh(mesh)

# %%
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# %%
grid.point_data["u"] = u_dimensionless_interpolated.x.array.real
grid.set_active_scalars("u")

# %%
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

# %%
plotter.bounds

# %%
# plotter.set_position([0,0,-1])

# %%
plotter.bounds

# %%
# plotter.set_scale(10,10,10)

# %%
# plotter.view_xy()

# %%
# plotter.camera_position = "xz"
# plotter.camera.azimuth = 45
# plotter.camera.elevation = 30
plotter.camera.zoom(20)

# %%
plotter.show()

# %%
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# %%
grid.point_data["c_H3Op"] = c_H3Op_dimensionless_interpolated.x.array.real
grid.set_active_scalars("c_H3Op")

# %%
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

# %%
plotter.camera.zoom(20)

# %%
plotter.show()

# %%
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# %%
grid.point_data["c_OHm"] = c_OHm_dimensionless_interpolated.x.array.real
grid.set_active_scalars("c_OHm")

# %%
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

# %%
plotter.camera.zoom(20)

# %%
plotter.show()

# %% [markdown]
# ## Surface integrals

# %% [markdown]
# ### Surface excess in 1d case

# %%
c

# %%
z

# %%
delta_u

# %%
debye_length

# %%
L = 10*debye_length

# %%
L

# %%
pnp_1d = PoissonNernstPlanckSystemFEniCSx(c=c, z=z, delta_u=delta_u, L=L, N=1000)

# %%
pnp_1d.use_standard_interface_bc()

# %%
potential_ref_unitless, c_ref_unitless, _ = pnp_1d.solve()

# %%
c_H3Op_ref_unitless = c_ref_unitless[0,:]
c_OHm_ref_unitless = c_ref_unitless[1,:]

# %%
x_ref_unitless = pnp_1d.grid_dimensionless

# %%
plt.plot(x_ref_unitless, c_H3Op_ref_unitless)
plt.plot(x_ref_unitless, c_OHm_ref_unitless)

# %%
plt.plot(x_ref_unitless, c_H3Op_ref_unitless-1)
plt.plot(x_ref_unitless, c_OHm_ref_unitless-1)

# %%
N_excess_H3Op_ref = np.trapz(c_H3Op_ref_unitless-1., x_ref_unitless)

# %%
N_excess_OHm_ref = np.trapz(c_OHm_ref_unitless-1., x_ref_unitless)

# %%
N_excess_H3Op_ref

# %%
N_excess_OHm_ref

# %%
dx = np.mean(x_smooth_dimensionless[1:]-x_smooth_dimensionless[:-1])

# %%
dx

# %%
x_smooth_dimensionless[0]

# %%
A_apparent_smooth = x_smooth_dimensionless[-1] - x_smooth_dimensionless[0] + dx

# %%
A_apparent_smooth

# %%
N_excess_OHm_ref*A_apparent_smooth

# %%
N_excess_H3Op_ref*A_apparent_smooth

# %% [markdown]
# ### Reference amount of species

# %%
reference_concentration_dimensionless = dolfinx.fem.Constant(mesh, default_scalar_type(1))

# %%
N_unit_concentration_expression = dolfinx.fem.form(reference_concentration_dimensionless*ufl.dx)

# %%
N_unit_concentration = dolfinx.fem.assemble_scalar(N_unit_concentration_expression)

# %%
N_unit_concentration

# %% [markdown]
# ### Amount of H3Op

# %%
N_H3Op_dimensionless_expression = dolfinx.fem.form(c_H3Op_dimensionless_interpolated * ufl.dx)

# %%
N_H3Op_dimensionless_local = dolfinx.fem.assemble_scalar(N_H3Op_dimensionless_expression)

# %%
N_H3Op_dimensionless_local

# %%
N_H3Op_dimensionless = mesh.comm.allreduce(N_H3Op_dimensionless_local, op=MPI.SUM)

# %%
N_H3Op_dimensionless

# %%
N_H3Op_excess = N_H3Op_dimensionless - N_unit_concentration

# %%
N_H3Op_excess

# %%
N_H3Op_excess / N_unit_concentration  # depletion per volume

# %%
dN_H3Op_smooth = N_H3Op_excess - N_excess_H3Op_ref*A_apparent_smooth

# %%
dN_H3Op_rough

# %% [markdown]
# ### Amount of OHm

# %%
N_OHm_dimensionless_expression = dolfinx.fem.form(c_OHm_dimensionless_interpolated * ufl.dx)

# %%
N_OHm_dimensionless_local = dolfinx.fem.assemble_scalar(N_OHm_dimensionless_expression)

# %%
N_OHm_dimensionless_local

# %%
N_OHm_dimensionless = mesh.comm.allreduce(N_OHm_dimensionless_local, op=MPI.SUM)

# %%
N_OHm_dimensionless

# %%
N_OHm_excess = N_OHm_dimensionless - N_unit_concentration

# %%
N_OHm_excess

# %%
N_OHm_excess / N_unit_concentration  # excess per volume

# %%
# excess compared to flat surface
dN_OHm_smooth = N_OHm_excess - N_excess_OHm_ref*A_apparent_smooth

# %%
dN_OHm_smooth

# %% [markdown]
# ## Line integrals

# %%
dx = np.mean(x_smooth_dimensionless[1:]-x_smooth_dimensionless[:-1])

y_mean = np.mean(y_smooth_dimensionless)
y_zero_aligned = y_smooth_dimensionless - y_mean
x_zero_aligned = x_smooth_dimensionless - x_smooth_dimensionless[0] + dx

x0 = 0
x1 = x_zero_aligned[-1] + dx

# %%
y_mean

# %%
A_apparent_smooth

# %%
tol = 0.0001
y_max = 10
N_points = 1001

c_ref = 1 # dimensionless

cH3Op_integrals = []
cOHm_integrals = []

dN_H3Op_integrals = []
dN_OHm_integrals = []

bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

for x, y_min in zip(x_zero_aligned, y_zero_aligned):
    y_grid = np.linspace(y_min + tol, y_max - tol, N_points)
    # dy = np.mean(y_grid[1:]-y_grid[:-1])

    points = np.zeros((3, N_points))
    points[0,:] = x
    points[1,:] = y_grid
    
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    cH3Op_values = c_H3Op_dimensionless_interpolated.eval(points_on_proc, cells).T
    cOHm_values = c_OHm_dimensionless_interpolated.eval(points_on_proc, cells).T

    # line integrals
    cH3Op_integral = np.trapz(y=cH3Op_values, x=y_grid)
    cOHm_integral = np.trapz(y=cOHm_values, x=y_grid)

    dN_cH3Op_excess_integral = np.trapz(y=cH3Op_values-c_ref, x=y_grid) - N_excess_H3Op_ref
    dN_OHm_excess_integral = np.trapz(y=cOHm_values-c_ref, x=y_grid) - N_excess_OHm_ref

    cH3Op_integrals.append(cH3Op_integral)
    cOHm_integrals.append(cOHm_integral)

    dN_H3Op_integrals.append(dN_cH3Op_excess_integral)
    dN_OHm_integrals.append(dN_OHm_excess_integral)

# %%
x_rough_dimensionless.shape

# %%
y_rough_dimensionless.shape

# %%
cH3Op_integrals = np.array(cH3Op_integrals).reshape(x_smooth_dimensionless.shape[0])

# %%
cOHm_integrals = np.array(cOHm_integrals).reshape(x_smooth_dimensionless.shape[0])

# %%
dN_H3Op_integrals = np.array(dN_H3Op_integrals).reshape(x_smooth_dimensionless.shape[0])

# %%
dN_OHm_integrals = np.array(dN_OHm_integrals).reshape(x_smooth_dimensionless.shape[0])

# %%
out_array = np.vstack([
    x_smooth_dimensionless, y_smooth_dimensionless, cH3Op_integrals, cOHm_integrals, dN_H3Op_integrals, dN_OHm_integrals]).T

# %%
np.savetxt('data/integrals/smooth_parallel_line_integrals.csv', 
           out_array,
           delimiter=',',
           header='x [Debye lengths], y [Debye lengths], dimensionless cH3Op integrated along y-direction, dimensionless cOHM integrated along y-direction, excess amount of H3Op compared to flat surface, excess amount of H3Op compared to flat surface')

# %% [markdown]
# ### cH3Op integrals

# %%
plt.plot(x_rough_dimensionless, cH3Op_integrals)

# %% [markdown]
# ### cOHm integrals

# %%
plt.plot(x_rough_dimensionless, cOHm_integrals)

# %% [markdown]
# ### H3Op excess 

# %%
plt.plot(x_smooth_dimensionless, y_smooth_dimensionless, label="profile")
plt.plot(x_smooth_dimensionless, dN_H3Op_integrals, linestyle=':', label="dN H3Op")
plt.legend()

# %% [markdown]
# ### OHm excess

# %%
plt.plot(x_smooth_dimensionless, y_smooth_dimensionless, label="profile")
plt.plot(x_smooth_dimensionless, dN_OHm_integrals, linestyle=':', label="dN OHm")
plt.legend()

# %% [markdown]
# ### Zooms

# %%
x_rough_dimensionless.shape

# %%
subset = slice(100,200)

# %%
plt.plot(x_smooth_dimensionless[subset], y_smooth_dimensionless[subset], label="profile")
plt.plot(x_smooth_dimensionless[subset], dN_OHm_integrals[subset], linestyle=':', label="dN OHm")
plt.legend()

# %%
plt.plot(x_smooth_dimensionless[subset], y_smooth_dimensionless[subset], label="profile")
plt.plot(x_smooth_dimensionless[subset], dN_H3Op_integrals[subset], linestyle=':', label="dN H3Op")
plt.legend()

# %%
subset = slice(300,400)

# %%
plt.plot(x_smooth_dimensionless[subset], y_smooth_dimensionless[subset], label="profile")
plt.plot(x_smooth_dimensionless[subset], dN_OHm_integrals[subset], linestyle=':', label="dN OHm")
plt.legend()

# %%
plt.plot(x_smooth_dimensionless[subset], y_smooth_dimensionless[subset], label="profile")
plt.plot(x_smooth_dimensionless[subset], dN_H3Op_integrals[subset], linestyle=':', label="dN H3Op")
plt.hlines(y=0,xmin=x_smooth_dimensionless[subset][0], xmax=x_smooth_dimensionless[subset][-1], linestyle='--', color='gray')

# %%
plt.plot(x_smooth_dimensionless[subset], y_smooth_dimensionless[subset], label="profile")
plt.plot(x_smooth_dimensionless[subset], dN_H3Op_integrals[subset], linestyle=':', label="dN H3Op")
plt.plot(x_smooth_dimensionless[subset], dN_OHm_integrals[subset], linestyle=':', label="dN OHm")
plt.hlines(y=0,xmin=x_smooth_dimensionless[subset][0], xmax=x_smooth_dimensionless[subset][-1], linestyle='--', color='gray')
plt.legend()

# %% [markdown]
# # Analysis of solution for smooth profile, vertical

# %% [markdown]
# ## Read solution from file

# %%
checkpoint_filename = 'data/checkpoint/rectangle_with_single_smooth_edge_vertical_mesh.bp'

# %%
mesh = adios4dolfinx.read_mesh(MPI.COMM_WORLD, checkpoint_filename, "BP4", dolfinx.mesh.GhostMode.none)

# %%
P = basix.ufl.element('Lagrange', mesh.basix_cell(), 3)
elements = [P] * 3
H = basix.ufl.mixed_element(elements)
W = dolfinx.fem.FunctionSpace(mesh, H)

# %%
w = dolfinx.fem.Function(W)

# %%
w.function_space.mesh

# %%
adios4dolfinx.read_function(w, checkpoint_filename)

# %%
w

# %%
u_function, c_H3Op_function, c_OHm_function = w.split()

# %%
gdim = mesh.geometry.dim

# %%
gdim

# %%
H0 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
W0 = FunctionSpace(mesh, H0)

# %%
u_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)
u_dimensionless_interpolated.interpolate(u_function)

# %%
c_H3Op_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)
c_H3Op_dimensionless_interpolated.interpolate(c_H3Op_function)

# %%
c_OHm_dimensionless_interpolated = Function(W0, dtype=default_scalar_type)
c_OHm_dimensionless_interpolated.interpolate(c_OHm_function)

# %% [markdown]
# ## Plot with pyvista

# %%
pyvista.start_xvfb()

# %%
topology, cell_types, geometry = plot.vtk_mesh(mesh)

# %%
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# %%
grid.point_data["u"] = u_dimensionless_interpolated.x.array.real
grid.set_active_scalars("u")

# %%
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

# %%
plotter.camera.zoom(20)

# %%
plotter.show()

# %%
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# %%
grid.point_data["c_H3Op"] = c_H3Op_dimensionless_interpolated.x.array.real
grid.set_active_scalars("c_H3Op")

# %%
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

# %%
plotter.camera.zoom(20)

# %%
plotter.show()

# %%
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# %%
grid.point_data["c_OHm"] = c_OHm_dimensionless_interpolated.x.array.real
grid.set_active_scalars("c_OHm")

# %%
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()

# %%
plotter.camera.zoom(20)

# %%
plotter.show()

# %% [markdown]
# ## Surface integrals

# %% [markdown]
# ### Surface excess in 1d case

# %%
c

# %%
z

# %%
delta_u

# %%
debye_length

# %%
L = 10*debye_length

# %%
L

# %%
pnp_1d = PoissonNernstPlanckSystemFEniCSx(c=c, z=z, delta_u=delta_u, L=L, N=1000)

# %%
pnp_1d.use_standard_interface_bc()

# %%
potential_ref_unitless, c_ref_unitless, _ = pnp_1d.solve()

# %%
c_H3Op_ref_unitless = c_ref_unitless[0,:]
c_OHm_ref_unitless = c_ref_unitless[1,:]

# %%
x_ref_unitless = pnp_1d.grid_dimensionless

# %%
plt.plot(x_ref_unitless, c_H3Op_ref_unitless)
plt.plot(x_ref_unitless, c_OHm_ref_unitless)

# %%
plt.plot(x_ref_unitless, c_H3Op_ref_unitless-1)
plt.plot(x_ref_unitless, c_OHm_ref_unitless-1)

# %%
N_excess_H3Op_ref = np.trapz(c_H3Op_ref_unitless-1., x_ref_unitless)

# %%
N_excess_OHm_ref = np.trapz(c_OHm_ref_unitless-1., x_ref_unitless)

# %%
N_excess_H3Op_ref

# %%
N_excess_OHm_ref

# %%
dx = np.mean(x_smooth_vert_dimensionless[1:]-x_smooth_vert_dimensionless[:-1])

# %%
dx

# %%
A_apparent_smooth_vertical = x_smooth_vert_dimensionless[-1] - x_smooth_vert_dimensionless[0] + dx

# %%
A_apparent_smooth_vertical

# %%
N_excess_OHm_ref*A_apparent_smooth_vertical

# %%
N_excess_H3Op_ref*A_apparent_smooth_vertical

# %% [markdown]
# ### Reference amount of species

# %%
reference_concentration_dimensionless = dolfinx.fem.Constant(mesh, default_scalar_type(1))

# %%
N_unit_concentration_expression = dolfinx.fem.form(reference_concentration_dimensionless*ufl.dx)

# %%
N_unit_concentration = dolfinx.fem.assemble_scalar(N_unit_concentration_expression)

# %%
N_unit_concentration

# %% [markdown]
# ### Amount of H3Op

# %%
N_H3Op_dimensionless_expression = dolfinx.fem.form(c_H3Op_dimensionless_interpolated * ufl.dx)

# %%
N_H3Op_dimensionless_local = dolfinx.fem.assemble_scalar(N_H3Op_dimensionless_expression)

# %%
N_H3Op_dimensionless_local

# %%
N_H3Op_dimensionless = mesh.comm.allreduce(N_H3Op_dimensionless_local, op=MPI.SUM)

# %%
N_H3Op_dimensionless

# %%
N_H3Op_excess = N_H3Op_dimensionless - N_unit_concentration

# %%
N_H3Op_excess

# %%
N_H3Op_excess / N_unit_concentration  # depletion per volume

# %%
dN_H3Op_smooth_vertical = N_H3Op_excess - N_excess_H3Op_ref*A_apparent_smooth_vertical

# %%
dN_H3Op_smooth_vertical

# %% [markdown]
# ### Amount of OHm

# %%
N_OHm_dimensionless_expression = dolfinx.fem.form(c_OHm_dimensionless_interpolated * ufl.dx)

# %%
N_OHm_dimensionless_local = dolfinx.fem.assemble_scalar(N_OHm_dimensionless_expression)

# %%
N_OHm_dimensionless_local

# %%
N_OHm_dimensionless = mesh.comm.allreduce(N_OHm_dimensionless_local, op=MPI.SUM)

# %%
N_OHm_dimensionless

# %%
N_OHm_excess = N_OHm_dimensionless - N_unit_concentration

# %%
N_OHm_excess

# %%
N_OHm_excess / N_unit_concentration  # excess per volume

# %%
# excess compared to flat surface
dN_OHm_smooth_vertical = N_OHm_excess - N_excess_OHm_ref*A_apparent_smooth_vertical

# %%
dN_OHm_smooth_vertical

# %% [markdown]
# ## Line integrals

# %%
dx = np.mean(x_smooth_vert_dimensionless[1:]-x_smooth_vert_dimensionless[:-1])

y_mean = np.mean(y_smooth_vert_dimensionless)
y_zero_aligned = y_smooth_vert_dimensionless - y_mean
x_zero_aligned = x_smooth_vert_dimensionless - x_smooth_vert_dimensionless[0] + dx

x0 = 0
x1 = x_zero_aligned[-1] + dx

# %%
y_mean

# %%
A_apparent_smooth_vertical

# %%
tol = 0.0001
y_max = 10
N_points = 1001

c_ref = 1 # dimensionless

cH3Op_integrals = []
cOHm_integrals = []

dN_H3Op_integrals = []
dN_OHm_integrals = []

bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

for x, y_min in zip(x_zero_aligned, y_zero_aligned):
    y_grid = np.linspace(y_min + tol, y_max - tol, N_points)
    # dy = np.mean(y_grid[1:]-y_grid[:-1])

    points = np.zeros((3, N_points))
    points[0,:] = x
    points[1,:] = y_grid
    
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    cH3Op_values = c_H3Op_dimensionless_interpolated.eval(points_on_proc, cells).T
    cOHm_values = c_OHm_dimensionless_interpolated.eval(points_on_proc, cells).T

    # line integrals
    cH3Op_integral = np.trapz(y=cH3Op_values, x=y_grid)
    cOHm_integral = np.trapz(y=cOHm_values, x=y_grid)

    dN_cH3Op_excess_integral = np.trapz(y=cH3Op_values-c_ref, x=y_grid) - N_excess_H3Op_ref
    dN_OHm_excess_integral = np.trapz(y=cOHm_values-c_ref, x=y_grid) - N_excess_OHm_ref

    cH3Op_integrals.append(cH3Op_integral)
    cOHm_integrals.append(cOHm_integral)

    dN_H3Op_integrals.append(dN_cH3Op_excess_integral)
    dN_OHm_integrals.append(dN_OHm_excess_integral)

# %%
cH3Op_integrals = np.array(cH3Op_integrals).reshape(x_rough_vert_dimensionless.shape[0])

# %%
cOHm_integrals = np.array(cOHm_integrals).reshape(x_rough_vert_dimensionless.shape[0])

# %%
dN_H3Op_integrals = np.array(dN_H3Op_integrals).reshape(x_rough_vert_dimensionless.shape[0])

# %%
dN_OHm_integrals = np.array(dN_OHm_integrals).reshape(x_rough_vert_dimensionless.shape[0])

# %%
out_array = np.vstack([
    x_rough_vert_dimensionless, y_rough_vert_dimensionless, cH3Op_integrals, cOHm_integrals, dN_H3Op_integrals, dN_OHm_integrals]).T

# %%
np.savetxt('data/integrals/smooth_vertical_line_integrals.csv', 
           out_array,
           delimiter=',',
           header='x [Debye lengths], y [Debye lengths], dimensionless cH3Op integrated along y-direction, dimensionless cOHM integrated along y-direction, excess amount of H3Op compared to flat surface, excess amount of H3Op compared to flat surface')

# %% [markdown]
# ### cH3Op integrals

# %%
plt.plot(x_smooth_vert_dimensionless, cH3Op_integrals)

# %% [markdown]
# ### cOHm integrals

# %%
plt.plot(x_smooth_vert_dimensionless, cOHm_integrals)

# %% [markdown]
# ### H3Op excess 

# %%
plt.plot(x_smooth_vert_dimensionless, y_smooth_vert_dimensionless, label="profile")
plt.plot(x_smooth_vert_dimensionless, dN_H3Op_integrals, linestyle=':', label="dN H3Op")
plt.legend()

# %% [markdown]
# ### OHm excess

# %%
plt.plot(x_smooth_vert_dimensionless, y_smooth_vert_dimensionless, label="profile")
plt.plot(x_smooth_vert_dimensionless, dN_OHm_integrals, linestyle=':', label="dN OHm")
plt.legend()

# %% [markdown]
# ### Zooms

# %%
subset = slice(100,200)

# %%
plt.plot(x_smooth_vert_dimensionless[subset], y_smooth_vert_dimensionless[subset], label="profile")
plt.plot(x_smooth_vert_dimensionless[subset], dN_OHm_integrals[subset], linestyle=':', label="dN OHm")
plt.legend()

# %%
plt.plot(x_smooth_vert_dimensionless[subset], y_smooth_vert_dimensionless[subset], label="profile")
plt.plot(x_smooth_vert_dimensionless[subset], dN_H3Op_integrals[subset], linestyle=':', label="dN H3Op")
plt.legend()

# %%
subset = slice(600,800)

# %%
plt.plot(x_smooth_vert_dimensionless[subset], y_smooth_vert_dimensionless[subset], label="profile")
plt.plot(x_smooth_vert_dimensionless[subset], dN_OHm_integrals[subset], linestyle=':', label="dN OHm")
plt.legend()

# %%
plt.plot(x_smooth_vert_dimensionless[subset], y_smooth_vert_dimensionless[subset], label="profile")
plt.plot(x_smooth_vert_dimensionless[subset], dN_H3Op_integrals[subset], linestyle=':', label="dN H3Op")
plt.hlines(y=0,xmin=x_smooth_vert_dimensionless[subset][0], xmax=x_smooth_vert_dimensionless[subset][-1], linestyle='--', color='gray')
plt.legend()

# %%
plt.plot(x_smooth_vert_dimensionless[subset], y_smooth_vert_dimensionless[subset], label="profile")
plt.plot(x_smooth_vert_dimensionless[subset], dN_H3Op_integrals[subset], linestyle=':', label="dN H3Op")
plt.plot(x_smooth_vert_dimensionless[subset], dN_OHm_integrals[subset], linestyle=':', label="dN OHm")
plt.hlines(y=0,xmin=x_smooth_vert_dimensionless[subset][0], xmax=x_smooth_vert_dimensionless[subset][-1], linestyle='--', color='gray')
plt.legend()

# %% [markdown]
# # Final comparison

# %%
import pandas as pd

# %%
df = pd.DataFrame(
    [[dN_H3Op_rough, dN_H3Op_rough_vertical, dN_H3Op_smooth, dN_H3Op_smooth_vertical],
     [dN_OHm_rough, dN_OHm_rough_vertical, dN_OHm_smooth, dN_OHm_smooth_vertical]],
    columns=["rough, parallel", "rough, vertical", "smooth, parallel", "smooth, vertical"],
    index=["H3Op", "OHm"])

# %%
df

# %%
df.to_csv("data/integrals/excess_dimensionless_summary.csv")

# %%
