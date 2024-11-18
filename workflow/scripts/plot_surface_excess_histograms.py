input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

csv_file = input.csv_file
# reference_csv_file = input.reference_csv_file
png_file = output.png_file
svg_file = output.svg_file

number_of_species = config["number_of_species"]

histogram_bins = config["histogram_bins"]

import logging
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the histogram data from CSV
hist_data = pd.read_csv(csv_file)

# Create a figure with subplots
fig, axes = plt.subplots(1, number_of_species, figsize=(6 * number_of_species, 5))

# Colors for each species (can be extended for more species)
colors = ['skyblue', 'lightgreen']

surface_excess_labels = [
    "$\Gamma_{\mathrm{H}_3\mathrm{O}^+} (c_\mathrm{bulk}\, \lambda_D)$",
    "$\Gamma_{\mathrm{OH}^-} (c_\mathrm{bulk}\, \lambda_D)$"
]

# Loop over species
for i in range(number_of_species):
    ax = axes[i] if number_of_species > 1 else axes  # Handle case of single subplot

    bin_edges = hist_data[f'bin_edges_{i}'].to_numpy()
    counts = hist_data[f'count_{i}'].to_numpy()

    logger.debug("bin edges: %s", bin_edges)
    logger.debug("count: %s", counts)

    dbin = bin_edges[1] - bin_edges[0]

    extended_bin_edges = np.array([*bin_edges, bin_edges[-1]+dbin])

    bin_centers = (extended_bin_edges[:-1] + extended_bin_edges[1:]) / 2
    data = np.repeat(bin_centers, counts.astype(int))

    logger.debug("data: %s", data)
    params = stats.skewnorm.fit(data)

    # Extract parameters for easier access
    a, loc, scale = params
    logger.debug("a: %s, loc: %s, scale: %s", a, loc, scale)

    # Step 3: Generate points for the fitted curve
    x = np.linspace(min(bin_centers), max(bin_centers), 100)
    pdf_fitted = stats.skewnorm.pdf(x, a, loc, scale)

    # ax.hist(data, density=True, bins=histogram_bins, histtype='stepfilled', alpha=0.7,
    #        edgecolor='black', color=colors[i])
    norm = np.sum(counts)*dbin
    logger.debug("norm: %s", norm)

    ax.bar(bin_edges, counts/norm,
           width=dbin,
           align='edge', alpha=0.7, color=colors[i], edgecolor='black')

    ax.plot(x, pdf_fitted, 'r-', lw=1, label='Fitted Skew-Normal Distribution')

    ax.set_title(f'Histogram of {surface_excess_labels[i]}')
    ax.set_xlabel(surface_excess_labels[i])
    ax.set_ylabel('Probability')

# Adjust layout and display the plot
fig.tight_layout()
fig.savefig(svg_file)
fig.savefig(png_file)
