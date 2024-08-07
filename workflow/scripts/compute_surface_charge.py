input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

reference_concentrations = config["reference_concentrations"]
number_charges = config["number_charges"]
number_of_species = config["number_of_species"]
temperature = config["temperature"]
relative_permittivity = config["relative_permittivity"]

checkpoint_bp = input.interpolated_solution_checkpoint_bp

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import json

import basix
import dolfinx
import ufl
import adios4dolfinx

import numpy as np
import scipy.constants as sc

from mpi4py import MPI

from utils import ionic_strength, lambda_D

faraday_constant = sc.value('Faraday constant')

I = ionic_strength(z=number_charges, c=reference_concentrations)

debye_length = lambda_D(ionic_strength=I, temperature=temperature, relative_permittivity=relative_permittivity)

mesh = adios4dolfinx.read_mesh(checkpoint_bp,
                               comm=MPI.COMM_WORLD,
                               engine="BP4", ghost_mode=dolfinx.mesh.GhostMode.none)

single_element_CG1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
scalar_function_space_CG1 = dolfinx.fem.functionspace(mesh, single_element_CG1)
potential_function = dolfinx.fem.Function(scalar_function_space_CG1,
                                          dtype=dolfinx.default_scalar_type)
# concentration_functions = [dolfinx.fem.Function(scalar_function_space_CG1,
#                                          dtype=dolfinx.default_scalar_type) for _ in range(number_of_species)]

adios4dolfinx.read_function(
        filename=checkpoint_bp,
        u=potential_function, name="solution_function_0")

meshtags = {}
for i in range(mesh.topology.dim + 1):
    meshtags[i] = adios4dolfinx.read_meshtags(filename=checkpoint_bp,
                                              mesh=mesh, meshtag_name=f"meshtags_{i}")

facet_markers = adios4dolfinx.read_meshtags(filename=checkpoint_bp,
                                            mesh=mesh, meshtag_name="facet_markers")
cell_markers = adios4dolfinx.read_meshtags(filename=checkpoint_bp,
                                           mesh=mesh, meshtag_name="cell_markers")

for i in range(mesh.topology.dim + 1):
    logger.debug("%d d tags: %s", i, np.unique(meshtags[i].values))
    logger.debug("number of %d d tags: %d", i, len(meshtags[i].values))

logger.debug("number of facet_markers: %d", len(facet_markers.values))
logger.debug("unique facet_markers: %s", np.unique(facet_markers.values))
logger.debug("number of cell_markers: %d", len(cell_markers.values))
logger.debug("unique cell_markers: %s", np.unique(cell_markers.values))

#facets = dolfinx.mesh.locate_entities_boundary(mesh, dim=(mesh.topology.dim - 1),
#                                               marker=neumann_boundary)
#facet_tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, facets, np.full_like(facets, 1, dtype=np.int32))

# rough boundary is tagged 4
facets = facet_markers.find(4)

logger.debug("facets with tag 4: %s", facets)
logger.debug("number of facets with tag 4: %s", len(facets))

# create new MeshTags object to mark rough boundary
facet_tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, facets, np.full_like(facets, 1, dtype=np.int32))

logger.debug("number of tagged facets for flux cevaluation: %s", len(facet_tags.values))

n = ufl.FacetNormal(mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
surface_charge_density = ufl.dot(ufl.grad(potential_function), n)
surface_charge_expression = dolfinx.fem.form(surface_charge_density * ds)
surface_charge_local = dolfinx.fem.assemble_scalar(surface_charge_expression)
# \begin{equation*}
#     \sigma^* = - 2 \nabla_* \phi^* \cdot \mathbf{\hat{n}^*}
# \end{equation*}
surface_charge = -2*mesh.comm.allreduce(surface_charge_local, op=MPI.SUM)

data = {
            'profile': wildcards.profile,
            'surface_charge': surface_charge,
       }

with open(output.json_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)
