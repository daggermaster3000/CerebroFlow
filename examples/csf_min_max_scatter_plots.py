import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
import datetime

# code for plotting scatter plot of max-min ventral and dorsal flow and saving as pdf
# Input csv file

pdf = PdfPages('scatter_plots.pdf')

# load data
data = pd.read_csv("Z:\\qfavey\\01_Experiments\\01_CSF_FLOW\\IMAGING_TESTS\\Successful_Images\\Inpp5e\\Analysis_1\\Inpp5e_csf_flow_results.csv")
n = len(data["name"])
fg, ax = plt.subplots()

# scatter plot colored by name
# plot points
for ext, min, name, col in zip(data["extremum"],data["minimum"],data["name"],cm.rainbow(np.linspace(0, 1, len(data["minimum"])))):
    ax.scatter(ext,["Ventral"],alpha=0.5,label=name,color=col)
    ax.scatter(min,["Dorsal"],alpha=0.5,color=col)

# plot means
ax.scatter(np.mean(data["extremum"]),["Ventral"], marker="|",color="red",alpha=0.5,s=500)
ax.scatter(np.mean(data["minimum"]),["Dorsal"], marker="|",color="red",alpha=0.5,s=500)


# add styling
ax.set_title(f"Extreme values of csf profiles\n n={n}.")
ax.set_xlabel("Velocity (um/s)")

ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.legend()

# plot styling
ax.minorticks_on()
width = 3
margin = (1 - width) + width / 2
ax.set_ylim(margin, 1.5)


# For the X Axis
ax.tick_params(
        axis="x",
    which="both",
    length=2,
    direction='in'
)
ax.tick_params(
        axis="x",
    which="major",
    length=5,
    width=2,
    direction='in'
)

# For the Y Axis
ax.tick_params(
        axis="y",
    which="major",
    length=2,
    width=50,
    labelrotation=45.0)

# For the Y Axis
ax.tick_params(
        axis="y",
    which="minor",
    length=0,
    width=0)
pdf.savefig(fg)
plt.close()
# -----------------------------------------------------------------------------------------------------------------------------------------
# scatter plot colored by condition
fg2, ax2 = plt.subplots()
#create the color map
colors_dict = {}
groups = data["group"]
colors = cm.rainbow(np.linspace(0, 1, len(set(groups))))
for group, color in zip(set(groups),colors):
    colors_dict[group] = color,data.loc[data["group"]==group,"extremum"],data.loc[data["group"]==group,"minimum"]

# plot points
for group in set(data["group"]):
    ax2.scatter(colors_dict[group][1],["Ventral"]*len(colors_dict[group][1]),alpha=0.5,color=colors_dict[group][0], marker = "o",linestyle='None', label = group)
    ax2.scatter(colors_dict[group][2],["Dorsal"]*len(colors_dict[group][2]),alpha=0.5,color=colors_dict[group][0], marker = "o",linestyle='None')

# plot means
ax2.scatter(np.mean(data["extremum"]),["Ventral"], marker="|",color="red",alpha=0.5,s=500)
ax2.scatter(np.mean(data["minimum"]),["Dorsal"], marker="|",color="red",alpha=0.5,s=500)

# Create legend handles manually


# Add the legend to the plot
ax2.legend()

# add styling
ax2.set_title(f"Extreme values of csf profiles\n n={n}.")
ax2.set_xlabel("Velocity (um/s)")

ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(True)
ax2.legend()

# plot styling
plt.minorticks_on()
width = 3
margin = (1 - width) + width / 2
ax2.set_ylim(margin, 1.5)

# For the X Axis
ax2.tick_params(
        axis="x",
    which="both",
    length=2,
    direction='in'
)
ax2.tick_params(
        axis="x",
    which="major",
    length=5,
    width=2,
    direction='in'
)

# For the Y Axis
ax2.tick_params(
        axis="y",
    which="major",
    length=2,
    width=50,
    labelrotation=45.0)

# For the Y Axis
ax2.tick_params(
        axis="y",
    which="minor",
    length=0,
    width=0)
#plt.savefig("scatter_plots.pdf")

pdf.savefig(fg2)
pdf.close()
plt.close()

