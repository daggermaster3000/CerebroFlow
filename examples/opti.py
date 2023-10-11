import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PySimpleGUI as sg
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import sys
sys.path.append('funcs')
import kymo as ky

# set output
pdf = PdfPages('Opti_test.pdf')

# get image
path = sg.popup_get_file("", no_window=True, default_extension=".tif")
print("Image path: ",path.replace("/","\\"))

# parms
pixel_size = 0.1625
exp1 = ky.Kymo(path.replace("/","\\"), pixel_size=pixel_size, frame_time=0.159)
filter_sizes = np.arange(10,21)
thresholds = np.arange(0.7,0.9,0.1)

# graphs parms
figure, axs = plt.subplots(len(filter_sizes),len(thresholds))
# tests
for x,f in enumerate(filter_sizes):
    for y,t in enumerate(thresholds):
        exp1.generate_kymo(threshold=t,dash=False, filter_size=f)
        _ , axs[x,y] = exp1.plot(save_profile=False, save_display=False,plot_create=True)
        axs[x,y].set_title(f'filter size: {f}, threshold: {t}')
plt.show()
pdf.savefig(figure)
       


pdf.close()
plt.close()

