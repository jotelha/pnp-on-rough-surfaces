input = snakemake.input
output = snakemake.output
config = snakemake.config
logfile = snakemake.log[0]

csv_file = input.csv_file
json_file = output.json_file

number_of_species = config["number_of_species"]

relative_lateral_cutoff = config["relative_lateral_cutoff"]

import logging

logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import json
import pandas as pd

df = pd.read_csv(csv_file)

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

line_integral_rolling_mean_window = config["line_integral_rolling_mean_window"]
line_integral_rolling_mean_window_std = config["line_integral_rolling_mean_window_std"]

df_smeared_out = df.rolling(window=line_integral_rolling_mean_window,
                            center=True, on="x", win_type="gaussian").mean(
                                std=line_integral_rolling_mean_window_std)

data = {}

for i in range(number_of_species):
    y_value_label = f"excess_concentration_integral_{i}"

    max_index = df[y_value_label].idxmax()
    min_index = df[y_value_label].idxmin()

    smeared_out_max_index = df_smeared_out[y_value_label].idxmax()
    smeared_out_min_index = df_smeared_out[y_value_label].idxmin()

    data.update({
        f'maximum_surface_excess_{i}': df.loc[max_index, y_value_label],
        f'position_of_maximum_surface_excess_{i}': df.loc[max_index, 'x'],
        f'minimum_surface_excess_{i}': df.loc[min_index, y_value_label],
        f'position_of_minimum_surface_excess_{i}': df.loc[min_index, 'x'],

        f'maximum_smeared_out_surface_excess_{i}': df_smeared_out.loc[smeared_out_max_index, y_value_label],
        f'position_of_maximum_smeared_out_surface_excess_{i}': df_smeared_out.loc[smeared_out_max_index, 'x'],
        f'minimum_smeared_out_surface_excess_{i}': df_smeared_out.loc[smeared_out_min_index, y_value_label],
        f'position_of_minimum_smeared_out_surface_excess_{i}': df_smeared_out.loc[smeared_out_min_index, 'x'],
    })

with open(output.json_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)