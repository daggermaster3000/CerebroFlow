from cerebroflow import kymo as ky
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import os as os
# Code to generate and save kymographs

# Prompt user to choose file
# (pub data: Z:\\qfavey\\01_Experiments\\01_CSF_FLOW\\PIPELINE_TEST\\BioProtocol_CSFflowMeasurement\\TestFiles\\FlowData\\WT5_2_cropped.tif)
path = sg.popup_get_file("", no_window=True, default_extension=".tif")
print("Image path: ",os.path.normpath(path))
# pixel size: 0.16250000000000003 or 0.189 (pub)
pixel_size = 0.1625
exp1 = ky.Kymo(os.path.normpath(path), pixel_size=pixel_size, frame_time=0.159)
exp1.generate_kymo(threshold=0.5,dash=True)
plt.imsave("raw_kymo_test2.png",exp1.raw_kymo[200:205],)

