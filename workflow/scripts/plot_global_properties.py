input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config

number_of_species = config["number_of_species"]

output_dir = output.output_dir

import os.path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


properties_per_species = [p.format(i)
    for i in range(number_of_species) for p in [
        "amount_of_substance_{}",
        "amount_of_substance_per_real_surface_area_{}",
        "amount_of_substance_per_apparent_surface_area_{}",
        "d_amount_of_substance_per_real_surface_area_{}",
        "d_amount_of_substance_per_apparent_surface_area_{}",

        "maximum_surface_excess_{}",
        "minimum_surface_excess_{}",

        "maximum_smeared_out_surface_excess_{}",
        "minimum_smeared_out_surface_excess_{}",

        "surface_excess_per_real_surface_area_{}",
        "surface_excess_per_apparent_surface_area_{}",

        "relative_surface_excess_per_real_surface_area_{}",
        "relative_surface_excess_per_apparent_surface_area_{}",

        "relative_maximum_surface_excess_{}",
        "relative_minimum_surface_excess_{}"
    ]
]

properties = [
    "charge",
    "surface_charge",
    "apparent_surface_area",
    "real_surface_area",
    "geometrical_roughness_factor",
    "gouy_chapman_capacitance_per_area",
    "capacitance",
    "capacitance_per_real_surface_area",
    "capacitance_per_apparent_surface_area",
    "roughness_function",
    "d_capacitance_per_real_surface_area",
    "d_capacitance_per_apparent_surface_area",
    "charge_SI",
    "apparent_surface_area_SI",
    "real_surface_area_SI",
    "capacitance_SI",
    "gouy_chapman_capacitance_per_area_SI",
    "capacitance_per_real_surface_area_SI",
    "capacitance_per_apparent_surface_area_SI",
    *properties_per_species
]

df = pd.read_csv(input.csv_file)

os.makedirs(output_dir, exist_ok=True)
for p in properties:
    # plots over potential:
    output_file = os.path.join(output_dir, f"potential_{p}.png")
    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size as needed
    sns_plot = sns.lineplot(data=df, x="potential_bias", y=p, hue="profile", style="profile", legend=True, ax=ax)
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    fig.savefig(output_file)

    # log log
    output_file = os.path.join(output_dir, f"potential_{p}_loglog.png")
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(output_file)

    # plots over rms roughness parameters

    # height
    output_file = os.path.join(output_dir, f"rms_height_SI_{p}.png")
    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size as needed
    sns_plot = sns.scatterplot(data=df, x="rms_height_SI", y=p, hue="potential_bias", style="profile", legend=True,
                               ax=ax)
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    fig.savefig(output_file)

    # slope
    output_file = os.path.join(output_dir, f"rms_slope_SI_{p}.png")
    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size as needed
    sns_plot = sns.scatterplot(data=df, x="rms_slope_SI", y=p, hue="potential_bias", style="profile", legend=True,
                               ax=ax)
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    fig.savefig(output_file)

    # curvature
    output_file = os.path.join(output_dir, f"rms_curvature_SI_{p}.png")
    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size as needed
    sns_plot = sns.scatterplot(data=df, x="rms_curvature_SI", y=p, hue="potential_bias", style="profile", legend=True,
                               ax=ax)
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    fig.savefig(output_file)