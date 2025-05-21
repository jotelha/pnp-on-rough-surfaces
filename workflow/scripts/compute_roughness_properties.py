input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

reference_concentrations = config["reference_concentrations"]
number_charges = config["number_charges"]
# number_of_species = config["number_of_species"]
temperature = config["temperature"]
relative_permittivity = config["relative_permittivity"]

profile_csv = input.profile_csv
profile_label = wildcards.profile
json_file = output.json_file

import logging
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import json
import numpy as np
import scipy.constants as sc

# from SurfaceTopography.IO.XYZ import read_csv
from SurfaceTopography.NonuniformLineScan import NonuniformLineScan

from utils import ionic_strength, lambda_D

faraday_constant = sc.value('Faraday constant')

I = ionic_strength(z=number_charges, c=reference_concentrations)

debye_length = lambda_D(ionic_strength=I, temperature=temperature, relative_permittivity=relative_permittivity)


# x, y = np.loadtxt(profile_csv,
#                           skiprows=profile_config["skiprows"],
#                           delimiter=profile_config["delimiter"],
#                           usecols=profile_config["usecols"],
#                           unpack=profile_config["unpack"],
#                           max_rows=profile_config["max_rows"])

# xy = np.array(xy_list)
# x = np.array(x_list)
# y = np.array(y_list)

# expect already normalized profile as input
data = np.loadtxt(profile_csv, delimiter=',')

x_normalized, y_normalized = data[:,0], data[:,1]

x = x_normalized*debye_length
y = y_normalized*debye_length

# WE NEED TO SWITCH TO SI UNITS BEFORE LINE_SCANNING:
# REASON: WE ARE GENERATING A DIMENIONLSS MESH!

line_scan = NonuniformLineScan(x, y, unit=config["unit"])
line_scan_SI = line_scan.to_unit('m')

data = {
            'profile': wildcards.profile,
            'rms_height_SI': line_scan_SI.rms_height_from_profile(),  # Rq
            #'mad_height_SI': line_scan_SI.mad_height(),
            'rms_slope_SI': line_scan_SI.rms_slope_from_profile(),
            'rms_curvature_SI': line_scan_SI.rms_curvature_from_profile()
       }

with open(output.json_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)
