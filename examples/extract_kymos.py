# Code to generate and save kymographs

import sys
sys.path.append('cerebroflow')
import kymo as ky
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import os as os
import PIL
from skimage import io
import numpy as np


# Prompt user to choose file
folder = sg.popup_get_folder("", no_window=True)
print(folder)

# create a new directory for the kymos
try:
    output_dir = "examples/kymographs2"
    os.mkdir(output_dir)
except:
    print("Already an existing output directory ")

# parameters:
pixel_size = 0.1625     # pixel size: 0.16250000000000003 for the olympus
frame_time = 0.159


for file in os.listdir(folder):
    path = os.path.normpath(os.path.join(folder,file))
    print("Image path: ",path)
    # create the experiment object
    exp1 = ky.Kymo(path, pixel_size=pixel_size, frame_time=frame_time)
    exp1.generate_kymo(threshold=0.7,dash=False)
    center = exp1.cc_location   # get the location of the center of the cc
    io.imsave(os.path.join(output_dir,f'{file}_kymo.tiff'),exp1.raw_kymo[center].astype(np.uint16))
