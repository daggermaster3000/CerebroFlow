import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PySimpleGUI as sg
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append('cerebroflow')
import kymo as ky
from scipy import stats

# set output
pdf = PdfPages('Opti_test.pdf')

# get image
path = sg.popup_get_file("", no_window=True, default_extension=".tif")
print("Image path: ",path.replace("/","\\"))

# parms
pixel_size = 0.1625
exp1 = ky.Kymo(path.replace("/","\\"), pixel_size=pixel_size, frame_time=0.159)
filter_sizes = np.arange(6,10,1)
thresholds = np.arange(0.5,0.9,0.1)
max = [0,""]


# tests
for x,f in enumerate(filter_sizes):
    for y,t in enumerate(thresholds):
        exp1.generate_kymo(threshold=t,dash=False, filter_size=f,gol_parms=(60,3))
        fig, axs = exp1.plot(save_profile=False, save_display=False,plot_create=True)
        
        if max[0] < np.max(exp1.mean_velocities):
            max[0] = np.max(exp1.mean_velocities)
            max[1] = f'{exp1.name.rstrip(".tif")} filter size: {f}, threshold: {t}'
        axs.set_title(f'{exp1.name.rstrip(".tif")} filter size: {f}, threshold: {t}')
        pdf.savefig(fig)
        del fig, axs
        plt.close()

print("bestmax: ", max[1])   
pdf.close()


