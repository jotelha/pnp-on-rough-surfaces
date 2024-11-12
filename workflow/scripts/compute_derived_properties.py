input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config
logfile = snakemake.log[0]

reference_concentrations = config["reference_concentrations"]
number_charges = config["number_charges"]
number_of_species = config["number_of_species"]
temperature = config["temperature"]
relative_permittivity = config["relative_permittivity"]

potential_bias_SI = float(wildcards.potential)

import json

import scipy.constants as sc

from utils import ionic_strength, lambda_D

I = ionic_strength(z=number_charges, c=reference_concentrations)

debye_length = lambda_D(ionic_strength=I, temperature=temperature, relative_permittivity=relative_permittivity)

gas_constant = sc.value('molar gas constant')
faraday_constant = sc.value('Faraday constant')

thermal_voltage = gas_constant * temperature / faraday_constant

potential_bias = potential_bias_SI / thermal_voltage

with open(input.surface_charge_json_file, 'r') as file:
    surface_charge_data = json.load(file)

with open(input.volume_integrals_json_file, 'r') as file:
    volume_integrals_data = json.load(file)

with open(input.profile_properties_json_file, 'r') as file:
    profile_properties_data = json.load(file)

with open(input.roughness_properties_json_file, 'r') as file:
    roughness_properties_data = json.load(file)

with open(input.surface_integrals_extrema_json_file, 'r') as file:
    surface_integrals_extrema_data = json.load(file)

data = {
    **surface_charge_data,
    **volume_integrals_data,
    **profile_properties_data,
    **roughness_properties_data,
    **surface_integrals_extrema_data
}

amount_of_substance_per_real_surface_area = []
amount_of_substance_per_apparent_surface_area = []

surface_excess_per_real_surface_area = []
surface_excess_per_apparent_surface_area = []
for i in range(number_of_species):
    amount_of_substance_per_real_surface_area.append(
        data[f'amount_of_substance_{i}']/data['real_surface_area'])
    amount_of_substance_per_apparent_surface_area.append(
        data[f'amount_of_substance_{i}'] / data['apparent_surface_area'])

    surface_excess_per_real_surface_area.append(
        data[f'surface_excess_{i}'] / data['real_surface_area'])
    surface_excess_per_apparent_surface_area.append(
        data[f'surface_excess_{i}'] / data['apparent_surface_area'])

capacitance = - data['charge'] / potential_bias
capacitance_per_real_surface_area = capacitance / data['real_surface_area']
capacitance_per_apparent_surface_area = capacitance / data['apparent_surface_area']

capacitance_SI = faraday_constant*I*debye_length**3/thermal_voltage * capacitance
capacitance_per_real_surface_area_SI = (faraday_constant*I*debye_length/thermal_voltage) * capacitance_per_real_surface_area
capacitance_per_apparent_surface_area_SI = (faraday_constant*I*debye_length/thermal_voltage) * capacitance_per_apparent_surface_area

roughness_function = capacitance_per_apparent_surface_area / data["gouy_chapman_capacitance_per_area"]

data.update({
    **{f'amount_of_substance_per_real_surface_area_{i}': amount_of_substance_per_real_surface_area[i] for i in range(number_of_species)},
    **{f'amount_of_substance_per_apparent_surface_area_{i}': amount_of_substance_per_apparent_surface_area[i] for i in range(number_of_species)},
    **{f'surface_excess_per_real_surface_area_{i}': surface_excess_per_real_surface_area[i] for i in
       range(number_of_species)},
    **{f'surface_excess_per_apparent_surface_area_{i}': surface_excess_per_apparent_surface_area[i] for i in
       range(number_of_species)},

    'capacitance': capacitance,
    'capacitance_per_real_surface_area': capacitance_per_real_surface_area,
    'capacitance_per_apparent_surface_area': capacitance_per_apparent_surface_area,
    'capacitance_SI': capacitance_SI,
    'capacitance_per_real_surface_area_SI': capacitance_per_real_surface_area_SI,
    'capacitance_per_apparent_surface_area_SI': capacitance_per_apparent_surface_area_SI,
    'roughness_function': roughness_function
})

with open(output.json_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)