input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

potential_bias_SI = config["potential_bias"]
reference_concentrations = config["reference_concentrations"]
number_charges = config["number_charges"]
number_of_species = config["number_of_species"]
temperature = config["temperature"]
relative_permittivity = config["relative_permittivity"]

height_normalized = config["height_normalized"]

import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

from matscipy.electrochemistry import PoissonNernstPlanckSystem

png_file = output.png_file
svg_file = output.svg_file

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

from utils import ionic_strength, lambda_D

I = ionic_strength(z=number_charges, c=reference_concentrations)

debye_length = lambda_D(ionic_strength=I, temperature=temperature, relative_permittivity=relative_permittivity)

length = height_normalized*debye_length

gas_constant = sc.value('molar gas constant')
faraday_constant = sc.value('Faraday constant')

thermal_voltage = gas_constant * temperature / faraday_constant

potential_bias = potential_bias_SI / thermal_voltage

pnp = PoissonNernstPlanckSystem(
    c=reference_concentrations,
    z=number_charges,
    L=length, delta_u=potential_bias)

pnp.use_standard_interface_bc()
ui, nij, _ = pnp.solve()

x = np.linspace(0, length, 100)

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

# %%
fig, (ax1,ax4) = plt.subplots(nrows=1,ncols=1)

ax1.axvline(x=debye_length, label='Debye length', color='grey', linestyle=':')
ax1.text(debye_length*1.02, 0.01, "Debye length", rotation=90)

ax1.plot(pnp.grid, pnp.potential, marker='', color='tab:red', label='potential, PNP', linewidth=1, linestyle='-')

ax2 = ax1.twinx()
ax2.plot(x, np.ones(x.shape)*reference_concentrations[0], label='bulk concentration', color='grey', linestyle=':')
ax2.text(10, reference_concentrations[0]*1.1, "bulk concentration")

ax2.plot(pnp.grid, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')

ax2.plot(pnp.grid, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')

ax3 = ax1.twinx()
# Offset the right spine of ax3.  The ticks and label have already been
# placed on the right by twinx above.
ax3.spines["right"].set_position(("axes", 1.1))

# Having been created by twinx, ax3 has its frame off, so the line of its
# detached spine is invisible. First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(ax3)

# Second, show the right spine.
ax3.spines["right"].set_visible(True)

ax3.plot(pnp.grid, pnp.charge_density, label='charge density, PNP', color='grey', linewidth=1, linestyle='-')

ax1.legend(loc='upper right',  bbox_to_anchor=(-0.1,1.02), fontsize=15)
ax2.legend(loc='center right', bbox_to_anchor=(-0.1,0.5),  fontsize=15)
ax3.legend(loc='lower right',  bbox_to_anchor=(-0.1,-0.02), fontsize=15)

fig.tight_layout()
fig.savefig(svg_file)
fig.savefig(png_file)