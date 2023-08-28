from funcs import kymo as ky
import PySimpleGUI as sg

# Prompt user to choose file
# (pub data: Z:\\qfavey\\01_Experiments\\01_CSF_FLOW\\PIPELINE_TEST\\BioProtocol_CSFflowMeasurement\\TestFiles\\FlowData\\WT5_2_cropped.tif)
path = sg.popup_get_file("", no_window=True, default_extension=".tif")
print("Image path: ",path.replace("/","\\"))

exp1 = ky.Kymo(path.replace("/","\\"), pixel_size=0.189, frame_time=0.1)

exp1.test_filter()
exp1.test_threshold()
exp1.generate_kymo(threshold=0.5)
