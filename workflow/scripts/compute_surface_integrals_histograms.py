input = snakemake.input
output = snakemake.output
config = snakemake.config
logfile = snakemake.log[0]

input_csv = input.input_csv
output_csv = output.output_csv

number_of_species = config["number_of_species"]

relative_lateral_cutoff = config["relative_lateral_cutoff"]

histogram_bins = config["histogram_bins"]


import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

df = pd.read_csv(input_csv)

xmin = df["x"].min()
xmax = df["x"].max()

logger.debug("Read profile with min, max x (%g, %g)", xmin, xmax)

lateral_span = (xmax - xmin)
discarded_dx = lateral_span * relative_lateral_cutoff

logger.debug("Discard relative %f portion of total lateral span %g: %g", relative_lateral_cutoff, lateral_span, discarded_dx)

lower_boundary = xmin + discarded_dx
upper_boundary = xmax - discarded_dx

logger.debug("Filter data by lower and upper boundary (%g, %g)", lower_boundary, upper_boundary)

df = df[(df["x"] > lower_boundary) & (df["x"] < upper_boundary)]

data_columns_dict = {}

# Define the number of bins for the histogram
num_bins = histogram_bins  # You can adjust this number as needed

# Generate histogram data for both columns

for i in range(number_of_species):
    y_value_label = f"excess_concentration_integral_{i}"

    bin_edge_label = f"bin_edges_{i}"
    count_label = f"count_{i}"

    hist, bin_edges = np.histogram(df[y_value_label], bins=num_bins)

    data_columns_dict.update({
        bin_edge_label: bin_edges[:-1],
        count_label: hist,
    })

out_df = pd.DataFrame(data_columns_dict)
out_df.to_csv(output_csv, index=False)