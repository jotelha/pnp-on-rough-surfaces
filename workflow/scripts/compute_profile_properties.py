input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

debye_length = config["debye_length"]
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

vacuum_permittivity = sc.epsilon_0
# gas_constant = sc.value('molar gas constant')
# faraday_constant = sc.value('Faraday constant')

x_input, y_input = np.loadtxt(profile_csv,
                          skiprows=profile_config["skiprows"],
                          delimiter=profile_config["delimiter"],
                          usecols=profile_config["usecols"],
                          unpack=profile_config["unpack"],
                          max_rows=profile_config["max_rows"])

x_dimensional = x_input*profile_config["xscale"]
y_dimensional = y_input*profile_config["xscale"]

dx = np.mean(x_dimensional[1:]-x_dimensional[:-1])

logger.info("dx: %g", dx)

y_mean = np.mean(y_dimensional)
logger.info("y_mean: %g", y_mean)

y_zero_aligned = y_dimensional - y_mean
x_zero_aligned = x_dimensional - x_dimensional[0] + dx

x0 = 0
x1 = x_zero_aligned[-1] + dx
logger.info("(x0, x1): (%g, %g)", x0, x1)


apparent_surface_area_SI = x1 - x0

x_diffs = x_zero_aligned[1:] - x_zero_aligned[:-1]
logger.info("max(x_diffs): %g", np.max(x_diffs))
logger.info("min(x_diffs): %g", np.min(x_diffs))
y_diffs = y_zero_aligned[1:] - y_zero_aligned[:-1]
logger.info("max(y_diffs): %g", np.max(y_diffs))
logger.info("min(y_diffs): %g", np.min(y_diffs))

segment_lengths = np.sqrt(np.square(x_diffs) + np.square(y_diffs))
logger.info("max(segment_lengths): %g", np.max(segment_lengths))
logger.info("min(segment_lengths): %g", np.min(segment_lengths))

real_surface_area_SI = np.sum(segment_lengths)

logger.info("real_surface_area_SI: %g", real_surface_area_SI)

geometrical_roughness_factor = real_surface_area_SI / apparent_surface_area_SI

logger.info("geometrical_roughness_factor: %g", geometrical_roughness_factor)

gouy_chapman_capacitance_SI = (
        relative_permittivity*vacuum_permittivity*real_surface_area_SI/debye_length)

logger.info("gouy_chapman_capacitance_SI: %g", gouy_chapman_capacitance_SI)

data = {
            'profile': wildcards.profile,
            'apparent_surface_area_SI': apparent_surface_area_SI,
            'real_surface_area_SI': real_surface_area_SI,
            'geometrical_roughness_factor': geometrical_roughness_factor,
            'gouy_chapman_capacitance_SI': gouy_chapman_capacitance_SI
       }

with open(output.json_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)
