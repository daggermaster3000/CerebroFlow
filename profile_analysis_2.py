import pandas as pd
import matplotlib.pyplot as plt
import os as os 
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def get_dv_axis(profile,thresh,pixel_size,bias=-15):
        """
        Finds the start of the central canal along the dv_axis based on a given threshold and bias

        Returns:
        -------
        an array
        """
        bias = -30
        warn=False
        try:
            dv_axis = np.arange(-(len(profile)-(len(profile)-np.argwhere(profile>thresh)[0][0]))-bias,len(profile)-np.argwhere(profile>thresh)[0][0]-bias)*pixel_size # find start of canal based on first speed over arbitrary threshold
        except:
            thresh = 0
            dv_axis = np.arange(-(len(profile)-(len(profile)-np.argwhere(profile>thresh)[0][0]))-bias,len(profile)-np.argwhere(profile>thresh)[0][0]-bias)*pixel_size 
            print("WARNING: Weird profile encountered. Origin will be set at first non-zero value.\nCheck the input image :p")
            warn = True
        return dv_axis,warn

# Settings ------------------
# 1. Add all the files to the same folder 
# 2. Add the condition and the color you want it to be.
#    The condition must be included in the file name
#
#       (Edit here)
#            |
#            v

dir = 'Denoising Data'                     # path to the directory
color_dic = {"denoised":"springgreen","raw":"slateblue"}   # add conditions here
pdf = PdfPages('Denoising comparison.pdf')

# create empty groups for later
groups = {}
for i in color_dic: 
    groups[i] = [] 

# read data
data={}
traces = []
names = []
for file in os.listdir(dir):
    try:
        names.append(file.rstrip(".csv"))
        df = pd.read_csv(os.path.join(dir,file), encoding='ascii')
        traces.append(df["mean_vels"].to_list())
        del df
        
    except:
        print("Non csv file was skipped")

# pad the traces to match the longest one
max_length = max(len(arr) for arr in traces)    # Find the maximum length of all arrays
traces = [np.pad(arr, (5, max_length - len(arr)+5), mode='constant', constant_values=0) for arr in traces]  # Pad each array to match the maximum length

# recalculate the d_v axis
for name, trace in zip(names,traces):
    df = {} # temp dic to create our df structure
    dv_axis = get_dv_axis(trace,0.3,0.1625,-25)[0]
    df["dv_axis"] = dv_axis
    df["trace"] = trace
    data[name] = pd.DataFrame(df)


# PLOT INDIVIDUAL CURVES --------------------------------------------------------------

# plot settings
fig, axs = plt.subplots()

for name in data:
    cond = ""
    for i in color_dic:
        if i in name:
            cond = i
    plt.plot(data[name]['dv_axis'],data[name]['trace'],alpha=0.6,label=name)
    plt.fill_between(data[name]['dv_axis'],data[name]['trace'],alpha=.1)

# plot custom
plt.suptitle("Individual Csf flow profiles")
plt.xlabel("Absolute Dorso-Ventral position [um]")
plt.ylabel("Rostro-Caudal Velocity [um/s]")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

#save as pdf
plt.show()
pdf.savefig(fig)
plt.close()

# PLOT EACH CURVE BUT COLORED BY CONDITION --------------------------------------
# plot settings
fig, axs = plt.subplots()

for name in data:
    cond = ""
    for i in color_dic:
        if i in name:
            cond = i
    plt.plot(data[name]['dv_axis'],data[name]['trace'],alpha=0.6,label=cond,color=color_dic[cond])
    plt.fill_between(data[name]['dv_axis'],data[name]['trace'],alpha=.1,color=color_dic[cond])

# plot custom
plt.suptitle("Individual Csf flow profiles")
plt.xlabel("Absolute Dorso-Ventral position [um]")
plt.ylabel("Rostro-Caudal Velocity [um/s]")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

#save as pdf
plt.show()
pdf.savefig(fig)
plt.close()


# PLOT MEAN CURVES --------------------------------------------------------------
# plot settings
fig, axs = plt.subplots()

# get all the traces in the same place
for name in data:
    cond = ""
    for i in color_dic:
        if i in name:
            cond = i

    groups[cond].append(data[name]["trace"].tolist())

# calculate the mean trace for each group

for group in groups:
    mean = np.mean(groups[group],axis=0)   # calculate the mean trace
    x = get_dv_axis(mean,0.2,0.1625)[0]
    plt.plot(x,mean,alpha=0.6,label=group,color=color_dic[group])
    plt.fill_between(x,mean,alpha=.2,color=color_dic[group])

# plot custom
plt.suptitle("Mean Csf flow profiles")
plt.xlabel("Absolute Dorso-Ventral position [um]")
plt.ylabel("Rostro-Caudal Velocity [um/s]")

plt.legend()
plt.gca().get_legend().set_title("")

#save as pdf
plt.show()
pdf.savefig(fig)
plt.close()


# PLOT MEAN CURVES with normalized dv-axis--------------------------------------------------------------
# plot settings
fig, axs = plt.subplots()

# get all the traces in the same place
for name in data:
    cond = ""
    for i in color_dic:
        if i in name:
            cond = i

    groups[cond].append(data[name]["trace"].tolist())

# calculate the mean trace for each group
max_x = 0
# for normalisation
for group in groups:
    mean = np.mean(groups[group],axis=0)   # calculate the mean trace
    x = get_dv_axis(mean,0.3,0.1625,-25)[0]  

    # get max x to normalize the cc width (end of the canal is defined where the trace gets lower than 0.1)
    if x[np.where(mean>0.001)[0][-1]] > max_x:
        max_x = x[np.where(mean>0.001)[0][-1]]

for group in groups:
    mean = np.mean(groups[group],axis=0)   # calculate the mean trace
    x = get_dv_axis(mean,0.2,0.1625)[0]
    plt.plot(x/max_x,mean,alpha=0.6,label=group,color=color_dic[group])
    plt.fill_between(x/max_x,mean,alpha=.2,color=color_dic[group])

# plot custom
plt.suptitle("Mean Csf flow profiles")
plt.xlabel("Relative Dorso-Ventral position [A.u.]")
plt.ylabel("Absolute Rostro-Caudal Velocity [um/s]")

plt.legend()
plt.gca().get_legend().set_title("")

#save as pdf
plt.show()
pdf.savefig(fig)
plt.close()




pdf.close()


