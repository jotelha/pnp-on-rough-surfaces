input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

number_of_species = config["number_of_species"]

import json

with open(input.json_file, 'r') as file:
    data = json.load(file)


with open(input.reference_json_file, 'r') as file:
    reference_data = json.load(file)

d_amount_of_substance_per_real_surface_area_SI = []
d_amount_of_substance_per_apparent_surface_area_SI = []
for i in range(number_of_species):
    d_amount_of_substance_per_real_surface_area_SI.append(
        data[f'amount_of_substance_per_real_surface_area_SI_{i}']
        - reference_data[f'amount_of_substance_per_real_surface_area_SI_{i}'])
    d_amount_of_substance_per_apparent_surface_area_SI.append(
        data[f'amount_of_substance_per_apparent_surface_area_SI_{i}']
        - reference_data[f'amount_of_substance_per_apparent_surface_area_SI_{i}'])

d_capacitance_per_real_surface_area_SI = data['capacitance_per_real_surface_area_SI'] - reference_data['capacitance_per_real_surface_area_SI']
d_capacitance_per_apparent_surface_area_SI = data['capacitance_per_apparent_surface_area_SI'] - reference_data['capacitance_per_apparent_surface_area_SI']

data.update({
    **{f'd_amount_of_substance_per_real_surface_area_SI_{i}': d_amount_of_substance_per_real_surface_area_SI[i] for i in range(number_of_species)},
    **{f'd_amount_of_substance_per_apparent_surface_area_SI_{i}': d_amount_of_substance_per_apparent_surface_area_SI[i] for i in range(number_of_species)},
    'd_capacitance_per_real_surface_area_SI': d_capacitance_per_real_surface_area_SI,
    'd_capacitance_per_apparent_surface_area_SI': d_capacitance_per_apparent_surface_area_SI,
})

with open(output.json_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)
