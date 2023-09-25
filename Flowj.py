from funcs import kymo as ky
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import matplotlib.lines as mlines

# Prompt user to choose file
# (pub data: Z:\\qfavey\\01_Experiments\\01_CSF_FLOW\\PIPELINE_TEST\\BioProtocol_CSFflowMeasurement\\TestFiles\\FlowData\\WT5_2_cropped.tif)
path = sg.popup_get_file("", no_window=True, default_extension=".tif")
print("Image path: ",path.replace("/","\\"))
# pixel size: 0.16250000000000003 or 0.189 (pub)
pixel_size = 0.189
exp1 = ky.Kymo(path.replace("/","\\"), pixel_size=pixel_size, frame_time=0.1)
exp1.generate_kymo(threshold=0.5,dash=True)
plt.imsave("raw_kymo_test2.png",exp1.raw_kymo[200],)

