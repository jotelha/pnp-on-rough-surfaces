input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

csv_file = input.csv_file
png_file = output.png_file
svg_file = output.svg_file

number_of_species = config["number_of_species"]

import logging
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

len_input = len(input)
logger.info("%d input files", len_input)
assert len_input == 3*number_of_species + 1
X_txt_list = input[0:number_of_species]
predicted_Y_txt_list = input[number_of_species:2*number_of_species]
predicted_variance_txt_list = input[2*number_of_species:3*number_of_species]

logger.info("X.txt file: %s", X_txt_list)
logger.info("predicted_Y.txt file: %s", predicted_Y_txt_list)
logger.info("predicted_variance.txt file: %s", predicted_variance_txt_list)

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

DATA_POINTS_ALPHA = 0.5
CONFIDENCE_INTERVAL_ALPHA = 0.05
WIDE_CONFIDENCE_INTERVAL_ALPHA = 0.05
NARROW_CONFIDENCE_INTERVAL_ALPHA = 0.5
SIMPLE_MEAN_ALPHA = 0.5
NARROWLY_SCATTERED_DATA_POINTS_ALPHA = 0.01
WIDELY_SCATTERED_DATA_POINTS_ALPHA = 0.05

DATA_POINTS_MARKER_SIZE = 2
WIDELY_SCATTERED_DATA_POINTS_MARKER_SIZE = 2
NARROWLY_SCATTERED_DATA_POINTS_MARKER_SIZE = 1

WIDELY_SCATTERED_DATA_POINTS_MARKER='x'
NARROWLY_SCATTERED_DATA_POINTS_MARKER='o'

ci_factor = 2. # confidence interval, 2 ~ 95 %

y_lims = [
    [-1.246, -1.243],
    [3.291, 3.299]
]

species_labels = ["$\mathrm{H}_3\mathrm{O}^+$", "$\mathrm{OH}^-$"]

y_axis_labels = [
    "$\Gamma_{\mathrm{H}_3\mathrm{O}^+} (c^\infty\, \lambda_D)$",
    "$\Gamma_{\mathrm{OH}^-} (c^\infty\, \lambda_D)$"
]

color_l = ['tab:orange', 'tab:blue']

df = pd.read_csv(input.csv_file)


line_integral_rolling_mean_window = config["line_integral_rolling_mean_window"]
line_integral_rolling_mean_window_std = config["line_integral_rolling_mean_window_std"]

df_smoothed = df.rolling(window=line_integral_rolling_mean_window,
                         center=True, on="x", win_type="gaussian").mean(std=line_integral_rolling_mean_window_std)

data = {}

for i in range(number_of_species):
    y_value_label = f"excess_concentration_integral_{i}"

    reference_X = df["x"].values
    reference_Y = df[y_value_label].values

    rolling_mean_X = df_smoothed["x"].values
    rolling_mean_Y = df_smoothed[y_value_label].values

    X = np.loadtxt(X_txt_list[i])
    Y = np.loadtxt(predicted_Y_txt_list[i])
    variance = np.loadtxt(predicted_variance_txt_list[i])
    stddev = np.sqrt(variance)

    data[i] = {
        'reference_X': reference_X,
        'reference_Y': reference_Y,
        'rolling_mean_X': rolling_mean_X,
        'rolling_mean_Y': rolling_mean_Y,
        'X': X,
        'Y': Y,
        'variance': variance,
        'stddev': stddev,
    }

fig, ax1 = plt.subplots(1,1, figsize=None)

twins = [ax1.twinx() for i in range(number_of_species)]
p_list = []

color = 'dimgray'
ax1.set_xlabel('x ($\lambda_D)$')
ax1.set_ylabel('h ($\lambda_D)$', color=color)

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
p1, = ax1.plot(df["x"], df["y"], color=color, linestyle=":", linewidth=1, label="roughness profile")

# confidence interval
for i in range(number_of_species):
    color = color_l[i]

    X = data[i]['X']
    Y = data[i]['Y']
    stddev = data[i]['stddev']

    twins[i].fill_between(
        X,
        (Y-ci_factor*stddev),
        (Y+ci_factor*stddev),
        alpha=CONFIDENCE_INTERVAL_ALPHA,
        label=f'{i}: GPR on surface excess $2\sigma$ confidence interval ',
        color=color)

# original data
for i in range(number_of_species):
    color = color_l[i]
    reference_X = data[i]['reference_X']
    reference_Y = data[i]['reference_Y']
    p, = twins[i].plot(reference_X, reference_Y,
            label=f'{i}: original data',
            marker='x', markersize=DATA_POINTS_MARKER_SIZE,
            color=color, linestyle='none', alpha=DATA_POINTS_ALPHA)
    p_list.append(p)

# rolling average
for i in range(number_of_species):
    color = color_l[i]
    rolling_mean_X = data[i]['rolling_mean_X']
    rolling_mean_Y = data[i]['rolling_mean_Y']
    p, = twins[i].plot(rolling_mean_X, rolling_mean_Y,
            label=f'{i}: original data rolling mean', color=color)
            #marker='x', markersize=DATA_POINTS_MARKER_SIZE,
            #color=color, alpha=DATA_POINTS_ALPHA) #, linestyle='none')
    p_list.append(p)

# # now simple mean of original data
# for prefix, color in zip(prefix_l, color_l):
#     mean_distance_d = data[prefix][secondary_prefix]['mean_distance_d']
#     mean_force_d = data[prefix][secondary_prefix]['mean_force_d']
#     ax.plot(mean_distance_d, mean_force_d,
#             label=f'{prefix}, simple mean', alpha=SIMPLE_MEAN_ALPHA,
#             color=color, linewidth=SIMPLE_MEAN_LINEWIDTH)

# now GPR models
for i in range(number_of_species):
    color = color_l[i]
    X = data[i]['X']
    Y = data[i]['Y']
    p, = twins[i].plot(X, Y,
                  label=f'{i}: GPR model for surface excess',
                  color=color, alpha=1, linewidth=THICK_LINEWIDTH, linestyle=(0, (1, 2)))
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

for i in range(number_of_species):
    twins[i].set_ylim(y_lims[i])

ax1.legend(handles=[p1, *p_list])

# ax.grid(which='major', axis='y', linewidth=GRID_LINEWIDTH)

fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(svg_file)
fig.savefig(png_file)
