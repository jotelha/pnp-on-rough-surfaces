input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

number_of_species = config["number_of_species"]
potential_bias = config["potential_bias"]

import json

with open(input.surface_integrals_json_file, 'r') as file:
    surface_integrals_data = json.load(file)

with open(input.profile_properties_json_file, 'r') as file:
    profile_properties_data = json.load(file)

data = {**surface_integrals_data, **profile_properties_data}

amount_of_substance_per_real_surface_area_SI = []
amount_of_substance_per_apparent_surface_area_SI = []
for i in range(number_of_species):
    amount_of_substance_per_real_surface_area_SI.append(data[f'amount_of_substance_SI_{i}']/data['real_surface_area_SI'])
    amount_of_substance_per_apparent_surface_area_SI.append(data[f'amount_of_substance_SI_{i}'] / data['apparent_surface_area_SI'])

capacitance_SI =  - data['charge_SI'] / potential_bias
capacitance_per_real_surface_area_SI = capacitance_SI / data['real_surface_area_SI']
capacitance_per_apparent_surface_area_SI = capacitance_SI / data['apparent_surface_area_SI']

roughness_function = capacitance_SI / data["gouy_chapman_capacitance_SI"]

data.update({
    **{f'amount_of_substance_per_real_surface_area_SI_{i}': amount_of_substance_per_real_surface_area_SI[i] for i in range(number_of_species)},
    **{f'amount_of_substance_per_apparent_surface_area_SI_{i}': amount_of_substance_per_apparent_surface_area_SI[i] for i in range(number_of_species)},
    'capacitance_SI': capacitance_SI,
    'capacitance_per_real_surface_area_SI': capacitance_per_real_surface_area_SI,
    'capacitance_per_apparent_surface_area_SI': capacitance_per_apparent_surface_area_SI,
    'roughness_function': roughness_function
})

with open(output.json_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)