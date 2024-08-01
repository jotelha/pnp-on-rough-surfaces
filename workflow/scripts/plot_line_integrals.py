input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
config = snakemake.config

number_of_species = config["number_of_species"]

concentration_labels = ["$\mathrm{H}_3\mathrm{O}^+$", "$\mathrm{OH}^-$"]

import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

default_cycler = cycler(color=mcolors.TABLEAU_COLORS.keys())
plt.rc('axes', prop_cycle=default_cycler)

df = pd.read_csv(input.csv_file)

fig, ax1 = plt.subplots()

twin1 = ax1.twinx()
twin2 = ax1.twinx()


color = 'dimgray'
ax1.set_xlabel('x ($\lambda_D)$')
ax1.set_ylabel('h ($\lambda_D)$', color=color)
p1, = ax1.plot(df["x"], df["y"], color=color, label="roughness profile")

color = 'tab:orange'
y_value = df[f"excess_concentration_integral_0"]
p2, = twin1.plot(df["x"], y_value.rolling(30).mean(), label=concentration_labels[0], color=color)

color = 'tab:blue'
y_value = df[f"excess_concentration_integral_1"]
p3, = twin2.plot(df["x"], y_value.rolling(30).mean(), label=concentration_labels[1], color=color)

ax1.tick_params(axis='y', labelcolor=color)

twin2.spines.right.set_position(("axes", 1.2))

twin1.set_ylabel("$\sigma_{\mathrm{H}_3\mathrm{O}^+} (c_{\mathrm{bulk}} \lambda_D^{-2})$")
twin2.set_ylabel("$\sigma_{\mathrm{OH}^-} (c_{\mathrm{bulk}} \lambda_D^{-2})$")

ax1.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())


tkw = dict(size=4, width=1.5)
ax1.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax1.tick_params(axis='x', **tkw)

ax1.legend(handles=[p1, p2, p3])

# fig.legend()
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(output.svg_file)
fig.savefig(output.png_file)