input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

profile_csv = output.profile_csv
amplitude = float(wildcards.amplitude)
frequency = float(wildcards.frequency)


import logging
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import numpy as np

# def x_array (length, start=0, N=1001):
#    return np.linspace(start, length, N)

# def create_rough_edge(length, frequency, amplitude, N= 1001):
#     x = x_array(length, N=N)
#     if frequency == 0:
#         y = np.zeros(x.shape)
#     else:
#         y = amplitude*np.cos(2*frequency*np.pi*x)
#     return y

if frequency != 0:
    period = 1 / frequency
else:  # use frequency == 0 as a marker for a flat surface of length 1
    period = 1

x = np.linspace(0, period, 1001)

if frequency == 0:  # use frequency == 0 as a marker for a flat surface of length 1
    y = np.zeros(x.shape)
else:
    y = amplitude * np.cos(2 * frequency * np.pi * x)


# x = x_array(2 * (1 / frequency))
# y = create_rough_edge(2 * (1 / frequency), frequency, amplitude)

data = np.column_stack((x,y))

np.savetxt(profile_csv, data, delimiter=',')
