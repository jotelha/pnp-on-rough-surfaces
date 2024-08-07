input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

potential_bias = config["potential_bias"]
reference_concentrations = config["reference_concentrations"]
number_charges = config["number_charges"]
number_of_species = config["number_of_species"]
temperature = config["temperature"]
relative_permittivity = config["relative_permittivity"]

profile_csv = input.profile_csv
profile_label = wildcards.profile
json_file = output.json_file

profile_config = config["profiles"][profile_label]

import logging
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import json
import numpy as np
import scipy.constants as sc

from utils import  ionic_strength, lambda_D

vacuum_permittivity = sc.epsilon_0
gas_constant = sc.value('molar gas constant')
faraday_constant = sc.value('Faraday constant')

I = ionic_strength(z=number_charges, c=reference_concentrations)

debye_length = lambda_D(ionic_strength=I, temperature=temperature,
                        relative_permittivity=relative_permittivity)

x_input, y_input = np.loadtxt(profile_csv,
                          skiprows=profile_config["skiprows"],
                          delimiter=profile_config["delimiter"],
                          usecols=profile_config["usecols"],
                          unpack=profile_config["unpack"],
                          max_rows=profile_config["max_rows"])

x_dimensional = x_input
y_dimensional = y_input

x_normalized = x_dimensional * profile_config["xscale"] / debye_length
y_normalized = y_dimensional * profile_config["yscale"] / debye_length

dx = np.mean(x_normalized[1:]-x_normalized[:-1])

logger.info("dx: %g", dx)

y_mean = np.mean(y_normalized)
logger.info("y_mean: %g", y_mean)

y_zero_aligned = y_normalized - y_mean
x_zero_aligned = x_normalized - x_normalized[0] + dx

x0 = 0
x1 = x_zero_aligned[-1] + dx
logger.info("(x0, x1): (%g, %g)", x0, x1)

x_extended = np.array([x0, *x_zero_aligned, x1])
y_extended = np.array([0, *y_zero_aligned, 0])

apparent_surface_area = x_extended[-1] - x_extended[0]
apparent_surface_area_SI = apparent_surface_area*debye_length
logger.info("apparent_surface_area: %g", apparent_surface_area)
logger.info("apparent_surface_area_SI: %g", apparent_surface_area_SI)

x_diffs = x_extended[1:] - x_extended[:-1]
logger.info("max(x_diffs): %g", np.max(x_diffs))
logger.info("min(x_diffs): %g", np.min(x_diffs))
y_diffs = y_extended[1:] - y_extended[:-1]
logger.info("max(y_diffs): %g", np.max(y_diffs))
logger.info("min(y_diffs): %g", np.min(y_diffs))

segment_lengths = np.sqrt(np.square(x_diffs) + np.square(y_diffs))
logger.info("max(segment_lengths): %g", np.max(segment_lengths))
logger.info("min(segment_lengths): %g", np.min(segment_lengths))

real_surface_area = np.sum(segment_lengths)
real_surface_area_SI = real_surface_area*debye_length

logger.info("real_surface_area: %g", real_surface_area)
logger.info("real_surface_area_SI: %g", real_surface_area_SI)

geometrical_roughness_factor = real_surface_area / apparent_surface_area

logger.info("geometrical_roughness_factor: %g", geometrical_roughness_factor)

gouy_chapman_capacitance_per_area_SI = (
        relative_permittivity*vacuum_permittivity/debye_length*np.cosh(
            faraday_constant*potential_bias/(2*gas_constant*temperature)))
logger.info("gouy_chapman_capacitance_per_area_SI: %g", gouy_chapman_capacitance_per_area_SI)

gouy_chapman_capacitance_per_area = (
        gas_constant * temperature / (np.square(faraday_constant) * I * debye_length)
        * gouy_chapman_capacitance_per_area_SI
)
logger.info("gouy_chapman_capacitance_per_area: %g", gouy_chapman_capacitance_per_area)

data = {
            'profile': wildcards.profile,
            'apparent_surface_area': apparent_surface_area,
            'apparent_surface_area_SI': apparent_surface_area_SI,
            'real_surface_area': real_surface_area,
            'real_surface_area_SI': real_surface_area_SI,
            'geometrical_roughness_factor': geometrical_roughness_factor,
            'gouy_chapman_capacitance_per_area': gouy_chapman_capacitance_per_area,
            'gouy_chapman_capacitance_per_area_SI': gouy_chapman_capacitance_per_area_SI
       }

with open(output.json_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)
