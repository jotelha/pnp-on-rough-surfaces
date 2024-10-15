input = snakemake.input
output = snakemake.output
config = snakemake.config

csv_file = input.csv_file
json_file = output.json_file

number_of_species = config["number_of_species"]

import json
import pandas as pd

df = pd.read_csv(csv_file)

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