from funcs import kymo as ky
import PySimpleGUI as sg

# Prompt user to choose file
# (pub data: Z:\\qfavey\\01_Experiments\\01_CSF_FLOW\\PIPELINE_TEST\\BioProtocol_CSFflowMeasurement\\TestFiles\\FlowData\\WT5_2_cropped.tif)
path = sg.popup_get_file("", no_window=True, default_extension=".tif")
print("Image path: ",path.replace("/","\\"))

# open time lapse
data,name = ky.open_tiff(path.replace("/","\\"))

# run kymo
ky.kymo1(data,name,wiener=False)

# test thresholds (needs some working on)
# ky.test_kymo_parms(data,name,wiener=False)
# test filters (TODO)
