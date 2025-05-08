input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

profile_csv = output.profile_csv
# profile_label = wildcards.profile
amplitude = float(wildcards.amplitude)
frequency = float(wildcards.frequency)

json_file = output.json_file

profile_config = config["profiles"][profile_label]

import logging
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import json
import numpy as np

def x_array (length, start=0, N=1001):
    return np.linspace(start, length, N)

def create_rough_edge(length, frequency, amplitude, N= 1001):
    x = x_array(length, N=N)
    if frequency == 0:
        y = np.zeros(x.shape)
    else:
        y = amplitude*np.cos(2*frequency*np.pi*x)
    return y

x = x_array(2 * (1 / frequency))
y = create_rough_edge(2 * (1 / frequency), frequency, amplitude)

'''from SurfaceTopography.IO.XYZ import read_csv
from SurfaceTopography.NonuniformLineScan import NonuniformLineScan

with open(profile_csv, 'r') as f:
    X_str, Y_str = read_csv(f,
                            usecols=profile_config["usecols"],
                            sep=profile_config["delimiter"],
                            skiprows=profile_config["skiprows"])

# remove invalid values
x_list = []
y_list = []
for i, (x_str, y_str) in enumerate(zip(X_str, Y_str)):
    try:
        x = float(x_str)
        y = float(y_str)
    except:
        logger.warning("No valid float on position %d: %s, %s", i, x_str, y_str)
        continue

    # xy_list.append(np.array([x,y]))
    x_list.append(x)
    y_list.append(y)

# xy = np.array(xy_list)
x = np.array(x_list)
y = np.array(y_list)

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
    json.dump(data, json_file, indent=4)'''
