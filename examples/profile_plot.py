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
pixel_size=0.1625
fg, ax = plt.subplots()

# get the x-axis
mean_velocities = [data['means'][i].replace('[',"").replace(']',"").split(',') for i in range(0,n)]


for profile in mean_velocities:
    profile = [float(i) for i in profile]
    dv_axis = np.arange(-(len(profile)-(len(profile)-np.nonzero(profile)[0][0])),len(profile)-np.nonzero(profile)[0][0])*pixel_size # find start of canal based on first non zero speed
    ax.plot(dv_axis,profile,alpha=0.6)   
ax.set_title(f"Inpp5e flow profile \nn={n}")

plt.show()


