from funcs import kymo as ky
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import matplotlib.lines as mlines

# Prompt user to choose file

path = sg.popup_get_file("", no_window=True, default_extension=".tif")
print("Image path: ",path.replace("/","\\"))
# pixel size: 0.16250000000000003 or 0.189 (pub)
pixel_size = 0.16250000000000003
exp1 = ky.Kymo(path.replace("/","\\"), pixel_size=pixel_size, frame_time=0.159)
exp1.generate_kymo(threshold=0.8,dash=False)
plt.close()
plt.clf()
max_vels = [np.max(vels) for vels in exp1.velocities]
min_vels = [np.min(vels) for vels in exp1.velocities]
dv_axis = np.arange(-(len(max_vels)-(len(max_vels)-np.nonzero(max_vels)[0][0])),len(max_vels)-np.nonzero(max_vels)[0][0])*pixel_size # find start of canal based on first non zero speed

for x,y in zip(dv_axis, exp1.velocities):
    try:
        length = len(y)
        
    except:
        length = 1
       
    plt.scatter([x]*length,y,alpha=0.5,label="velocities", marker = "o",linestyle='None')
    plt.scatter([x],np.mean(y),marker="x",color="black",label="mean velocity",linestyle='None')
  
# Create legend handles manually

dots = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=5, label='Measured velocity',alpha=0.5)
crosses = mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=5, label='Mean velocity')

# Add the legend to the plot
plt.legend(handles=[dots, crosses])

plt.xlabel("Absolute d-v position (um)")
plt.ylabel("velocity (um/s)")
plt.xlim(0,9)
plt.suptitle("Velocity scatter plot")
# Save the plot to an image file

plt.savefig("velocity_scatter_plot.png")

# Close the Matplotlib figure to release resources
plt.close()
print("done")

#exp1.test_filter()
#exp1.test_threshold()


