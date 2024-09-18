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

height_normalized = config["height_normalized"]

potential_bias_SI = wildcards.potential

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

length = 0.5*height_normalized*debye_length

gas_constant = sc.value('molar gas constant')
faraday_constant = sc.value('Faraday constant')

thermal_voltage = gas_constant * temperature / faraday_constant

potential_bias = potential_bias_SI / thermal_voltage

pnp = PoissonNernstPlanckSystem(
    c=reference_concentrations,
    z=number_charges,
    L=length, delta_u=potential_bias_SI)

pnp.use_standard_interface_bc()
ui, nij, _ = pnp.solve()

x = np.linspace(0, length, 100)

fig, ax1 = plt.subplots(nrows=1,ncols=1)

ax1.axvline(x=debye_length, label='Debye length', color='grey', linestyle=':', linewidth=0.5)
# ax1.text(debye_length*1.02, 0.01, "Debye length", rotation=90)

p_potential, = ax1.plot(pnp.grid, pnp.potential, marker='', color='tab:red', label='potential, PNP', linewidth=1, linestyle='-')

ax2 = ax1.twinx()
ax2.plot(x, np.ones(x.shape)*reference_concentrations[0], label='bulk concentration', color='grey', linestyle=':',  linewidth=0.5)
ax2.text(10, reference_concentrations[0]*1.1, "bulk concentration")

ax2.plot(pnp.grid, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=1.5, linestyle=':')
ax2.plot(pnp.grid, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=1.5, linestyle=':')

tkw = dict(size=4, width=1.5)
ax1.tick_params(axis='y', colors=p_potential.get_color(), **tkw)
ax1.tick_params(axis='x', **tkw)

ax1.yaxis.label.set_color(p_potential.get_color())

ax1.set_xlabel("Height")
ax1.set_ylabel("Potential")
ax2.set_ylabel("Concentration")

ax1.set_xticks(
    ticks=[0, debye_length],
    labels=[0, "$\lambda_D$"]
)

ax1.set_yticks(
    ticks=[0, thermal_voltage],
    labels=[0, "$U_T$"]
)

ax2.set_yticks(
    ticks=[0, reference_concentrations[0]],
    labels=[0, "$c_\mathrm{bulk}$"]
)

fig.set_size_inches(2, 1.6)

fig.subplots_adjust(bottom=0.3, left=0.25, right=0.7)

fig.tight_layout()
fig.savefig(svg_file)
fig.savefig(png_file)