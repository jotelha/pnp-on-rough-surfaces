input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

csv_file = input.csv_file
reference_csv_file = input.reference_csv_file
png_file = output.png_file
svg_file = output.svg_file

number_of_species = config["number_of_species"]

import logging
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               ScalarFormatter)

DEFAULT_LINEWIDTH = 0.5
THIN_LINEWIDTH = 0.25
THICK_LINEWIDTH = 1.5
SIMPLE_MEAN_LINEWIDTH = 0.25
GRID_LINEWIDTH = 0.5

DATA_POINTS_ALPHA = 0.2
CONFIDENCE_INTERVAL_ALPHA = 0.05
WIDE_CONFIDENCE_INTERVAL_ALPHA = 0.05
NARROW_CONFIDENCE_INTERVAL_ALPHA = 0.5
SIMPLE_MEAN_ALPHA = 0.5
NARROWLY_SCATTERED_DATA_POINTS_ALPHA = 0.01
WIDELY_SCATTERED_DATA_POINTS_ALPHA = 0.05

DATA_POINTS_MARKER_SIZE = 4
WIDELY_SCATTERED_DATA_POINTS_MARKER_SIZE = 2
NARROWLY_SCATTERED_DATA_POINTS_MARKER_SIZE = 1

WIDELY_SCATTERED_DATA_POINTS_MARKER='x'
NARROWLY_SCATTERED_DATA_POINTS_MARKER='o'

ci_factor = 2. # confidence interval, 2 ~ 95 %

x_lim = [170, 350]

try:
    h_lim = config["profiles"][wildcards.profile]["surface_excess_local"]["hlim"]
except KeyError:
    logger.warning("No explicit hlim set for %s. Use global setting.", wildcards.profile)
    h_lim = config["surface_excess_local"]["hlim"]

try:
    h_ticks = config["profiles"][wildcards.profile]["surface_excess_local"]["hticks"]
except KeyError:
    logger.warning("No explicit hticks set for %s. Use global setting.", wildcards.profile)
    h_ticks = config["surface_excess_local"]["hticks"]

try:
    y_lims = config["profiles"][wildcards.profile]["surface_excess_local"]["ylims"]
except KeyError:
    logger.warning("No explicit ylims set for %s. Use global setting.", wildcards.profile)
    y_lims = config["surface_excess_local"]["ylims"]

try:
    y_ticks = config["profiles"][wildcards.profile]["surface_excess_local"]["yticks"]
except KeyError:
    logger.warning("No explicit yticks set for %s. Use global setting.", wildcards.profile)
    y_ticks = config["surface_excess_local"]["yticks"]


species_labels = ["$\mathrm{H}_3\mathrm{O}^+$", "$\mathrm{OH}^-$"]

x_axis_label = 'length x ($\lambda_D)$'
h_axis_label = 'profile height h ($\lambda_D)$'

y_axis_labels = [
    "$\Gamma_{\mathrm{H}_3\mathrm{O}^+} (c_\mathrm{bulk}\, \lambda_D)$",
    "$\Gamma_{\mathrm{OH}^-} (c_\mathrm{bulk}\, \lambda_D)$"
]

color_l = ['tab:orange', 'tab:blue']

df = pd.read_csv(csv_file)
reference_df = pd.read_csv(reference_csv_file)

line_integral_rolling_mean_window = config["line_integral_rolling_mean_window"]
line_integral_rolling_mean_window_std = config["line_integral_rolling_mean_window_std"]

df_smoothed = df.rolling(window=line_integral_rolling_mean_window,
                         center=True, on="x", win_type="gaussian").mean(
    std=line_integral_rolling_mean_window_std)

data = {}

for i in range(number_of_species):
    y_value_label = f"excess_concentration_integral_{i}"

    reference_X = reference_df["x"].values
    reference_Y = reference_df[y_value_label].values

    original_X = df["x"].values
    original_Y = df[y_value_label].values


    rolling_mean_X = df_smoothed["x"].values
    rolling_mean_Y = df_smoothed[y_value_label].values


    data[i] = {
        'reference_X': reference_X,
        'reference_Y': reference_Y,

        'original_X': original_X,
        'original_Y': original_Y,

        'rolling_mean_X': rolling_mean_X,
        'rolling_mean_Y': rolling_mean_Y,
    }

fig, ax1 = plt.subplots(1,1, figsize=None)

twins = [ax1.twinx() for i in range(number_of_species)]
p_list = []

color = 'dimgray'
ax1.set_xlabel(x_axis_label)
ax1.set_ylabel(h_axis_label, color=color)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
# ax1.yaxis.set_major_locator(MultipleLocator(10))
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
#
# # For the minor ticks, use no labels; default NullFormatter.
# ax1.yaxis.set_minor_locator(MultipleLocator(5))
#
# ax1.xaxis.set_major_locator(MultipleLocator(10))
# ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#
# # For the minor ticks, use no labels; default NullFormatter.
# ax1.xaxis.set_minor_locator(MultipleLocator(1))

# plot roughness profile
p1, = ax1.plot(
    df["x"], df["y"],
    color=color, linestyle="-", linewidth=1, label="roughness profile")

# reference data
for i in range(number_of_species):
    color = color_l[i]
    X = data[i]['reference_X']
    Y = data[i]['reference_Y']
    p, = twins[i].plot(X, Y,
            label=f'{i}: surface excess on flat surface', color=color,
            linestyle='--', linewidth=2, alpha=DATA_POINTS_ALPHA)
    p_list.append(p)

# mean
# for i in range(number_of_species):
#     color = color_l[i]
#     X = data[i]['original_X']
#     Y = data[i]['original_Y']
#     mean_Y = np.mean(Y) * np.ones(Y.shape)
#     p, = twins[i].plot(X, mean_Y,
#             label=f'{i}: original data mean', color=color,
#             linestyle='--', linewidth=1, alpha=DATA_POINTS_ALPHA)
#     p_list.append(p)

# original data
for i in range(number_of_species):
    color = color_l[i]
    X = data[i]['original_X']
    Y = data[i]['original_Y']
    p, = twins[i].plot(X, Y,
                       label=f'{i}: original data',
                       color=color,
                       alpha=DATA_POINTS_ALPHA, linewidth=0.5)
    # p, = twins[i].plot(X[::2], Y[::2],
    #         label=f'{i}: original data',
    #         marker='x', markersize=DATA_POINTS_MARKER_SIZE,
    #         color=color, linestyle='none', alpha=DATA_POINTS_ALPHA)
    p_list.append(p)

# rolling mean
for i in range(number_of_species):
    color = color_l[i]
    rolling_mean_X = data[i]['rolling_mean_X']
    rolling_mean_Y = data[i]['rolling_mean_Y']
    p, = twins[i].plot(rolling_mean_X, rolling_mean_Y,
            label=f'{i}: original data rolling mean', color=color,
            linestyle=(0, (1, 2)))
    p_list.append(p)

ax1.tick_params(axis='y', labelcolor=color)

for i in range(1, number_of_species):
    twins[i].spines.right.set_position(("axes", 1.0 + 0.2*i))

ax1.yaxis.label.set_color(p1.get_color())

for i in range(number_of_species):
    twins[i].set_ylabel(y_axis_labels[i])
    twins[i].yaxis.label.set_color(p_list[i].get_color())
    twins[i].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

tkw = dict(size=4, width=1.5)
ax1.tick_params(axis='y', colors=p1.get_color(), **tkw)

for i in range(number_of_species):
    twins[i].tick_params(axis='y', colors=p_list[i].get_color(), **tkw)

ax1.tick_params(axis='x', **tkw)

ax1.set_xlim(x_lim)
ax1.set_ylim(h_lim)
ax1.set_yticks(h_ticks)

for i in range(number_of_species):
    twins[i].set_ylim(y_lims[i])
    twins[i].set_yticks(y_ticks[i])

# ax1.legend(handles=[p1, *p_list])

# ax.grid(which='major', axis='y', linewidth=GRID_LINEWIDTH)

fig.set_size_inches(7, 2.5)
#  7.0, 5.25
fig.tight_layout()
fig.savefig(svg_file)
fig.savefig(png_file)
