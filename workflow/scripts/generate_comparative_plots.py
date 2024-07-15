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
        "amount_of_substance_SI_{}",
        "amount_of_substance_per_real_surface_area_SI_{}",
        "amount_of_substance_per_apparent_surface_area_SI_{}",
        "d_amount_of_substance_per_real_surface_area_SI_{}",
        "d_amount_of_substance_per_apparent_surface_area_SI_{}"]
]

properties = [
    "charge_SI",
    "apparent_surface_area_SI",
    "real_surface_area_SI",
    "geometrical_roughness_factor",
    "gouy_chapman_capacitance_SI",
    "capacitance_SI",
    "capacitance_per_real_surface_area_SI",
    "capacitance_per_apparent_surface_area_SI",
    "roughness_function",
    "d_capacitance_per_real_surface_area_SI",
    "d_capacitance_per_apparent_surface_area_SI",
    *properties_per_species
]

df = pd.read_csv(input.csv_file)

os.makedirs(output_dir, exist_ok=True)
for p in properties:
    output_file = os.path.join(output_dir, f"{p}.png")
    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size as needed
    sns_plot = sns.scatterplot(data=df, x="profile", y=p, hue="profile", style="profile", legend=False, ax=ax)
    sns_plot.set_xticklabels(sns_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    # fig = sns_plot.get_figure()
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    fig.savefig(output_file)