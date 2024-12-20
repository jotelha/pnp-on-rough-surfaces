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

d_amount_of_substance_per_real_surface_area = []
d_amount_of_substance_per_apparent_surface_area = []

relative_surface_excess_per_real_surface_area = []
relative_surface_excess_per_apparent_surface_area = []

relative_maximum_surface_excess = []
relative_minimum_surface_excess = []
for i in range(number_of_species):
    d_amount_of_substance_per_real_surface_area.append(
        data[f'amount_of_substance_per_real_surface_area_{i}']
        - reference_data[f'amount_of_substance_per_real_surface_area_{i}'])
    d_amount_of_substance_per_apparent_surface_area.append(
        data[f'amount_of_substance_per_apparent_surface_area_{i}']
        - reference_data[f'amount_of_substance_per_apparent_surface_area_{i}'])

    relative_surface_excess_per_real_surface_area.append(data[f'surface_excess_per_real_surface_area_{i}']
                                                     / reference_data[f'surface_excess_per_real_surface_area_{i}'])
    relative_surface_excess_per_apparent_surface_area.append(data[f'surface_excess_per_apparent_surface_area_{i}']
                                                         / reference_data[f'surface_excess_per_apparent_surface_area_{i}'])

    relative_maximum_surface_excess.append(data[f'maximum_surface_excess_{i}']
                                                     / reference_data[f'surface_excess_per_apparent_surface_area_{i}'])
    relative_minimum_surface_excess.append(data[f'minimum_surface_excess_{i}']
                                                         / reference_data[
                                                             f'surface_excess_per_apparent_surface_area_{i}'])


d_capacitance_per_real_surface_area = data['capacitance_per_real_surface_area'] - reference_data['capacitance_per_real_surface_area']
d_capacitance_per_apparent_surface_area = data['capacitance_per_apparent_surface_area'] - reference_data['capacitance_per_apparent_surface_area']


data.update({
    **{f'd_amount_of_substance_per_real_surface_area_{i}': d_amount_of_substance_per_real_surface_area[i] for i in range(number_of_species)},
    **{f'd_amount_of_substance_per_apparent_surface_area_{i}': d_amount_of_substance_per_apparent_surface_area[i] for i in range(number_of_species)},
    **{f'relative_surface_excess_per_real_surface_area_{i}': relative_surface_excess_per_real_surface_area[i] for i in range(number_of_species)},
    **{f'relative_surface_excess_per_apparent_surface_area_{i}': relative_surface_excess_per_apparent_surface_area[i] for i in range(number_of_species)},
    **{f'relative_maximum_surface_excess_{i}': relative_maximum_surface_excess[i] for i in range(number_of_species)},
    **{f'relative_minimum_surface_excess_{i}': relative_minimum_surface_excess[i] for i in range(number_of_species)},
    'd_capacitance_per_real_surface_area': d_capacitance_per_real_surface_area,
    'd_capacitance_per_apparent_surface_area': d_capacitance_per_apparent_surface_area,
})

with open(output.json_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)
