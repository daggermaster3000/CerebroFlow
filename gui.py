import PySimpleGUI as sg
from funcs import kymo as ky
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import threading


class GUI:
    def __init__(self):
        # Define the layout of the GUI
        self.layout = [
            [sg.Text("CSF Flow Analysis", font=("Helvetica", 20))],
            [sg.Text("File Path:"), sg.InputText(key="image_path"), sg.FilesBrowse()],
            [sg.Text("Output Folder:"), sg.InputText(key="output_path"), sg.FolderBrowse()],
            [sg.Column([
            [sg.Text("Pixel Size (um):"), sg.InputText(key="pixel_size", size=(6,2), default_text = 0.189)],
            [sg.Text("Frame Time (s):"), sg.InputText(key="frame_time", size=(6,2), default_text = 0.1)],
            [sg.Text("Filter size (px):"), sg.InputText(key="filter_size", size=(6,2), default_text = None)],
            [sg.Text("Threshold:"), sg.InputText(key="threshold", size=(6,2), default_text = 0.5)],
            [sg.Text("Thresholding Method:")],
            [sg.Radio("Hardcore", "thresholding", key="method_hardcore"),
            sg.Radio("Quantile", "thresholding", key="method_quantile", default=True)],
            [sg.Text("Naming Method:")],
            [sg.Radio("Filename", "naming_method", key="Filename")],
            [sg.Radio("Custom", "naming_method", key="Custom", default=True),
            sg.Text("Group name:"), sg.InputText(key="group_name", default_text = "GroupName")],
            [sg.Text("Outputs:")],
            [sg.Checkbox("Individual flow profiles", key="individual_profiles", default=True)],
            [sg.Checkbox("Total flow profile", key="total_profile", default=True)],
            [sg.Checkbox("CSV Data Table", key="csv_table", default=True)],
            [sg.Button("Test threshold"), sg.Button("Test filter")],
            [sg.Button("Run Analysis"), sg.Button("Exit")],
            ], element_justification='left')],
        ]
        # Create the window
        self.window = sg.Window("CSF Flow Analysis GUI", self.layout, element_justification="center")
        self.analysis_running = False

    def test_threshold(self):
                    
        pixel_size = float(self.values["pixel_size"])
        frame_time = float(self.values["frame_time"])
        path = sg.popup_get_file("", no_window=True, default_extension=".tif")
        exp = ky.Kymo(path.replace("/","\\"), pixel_size=pixel_size, frame_time=frame_time)
        exp.test_threshold()

    def test_filter(self):
                    
        pixel_size = float(self.values["pixel_size"])
        frame_time = float(self.values["frame_time"])
        path = sg.popup_get_file("", no_window=True, default_extension=".tif")
        exp = ky.Kymo(path.replace("/","\\"), pixel_size=pixel_size, frame_time=frame_time)
        exp.test_filter()


    def start(self):
        # Event loop
        while True:
            self.event, self.values = self.window.read()

            if self.event == sg.WIN_CLOSED or self.event == "Exit":
                break
            
            elif self.event == "Run Analysis":
                if self.analysis_running:
                    sg.popup("Analysis in progress please wait...", title="CSF Flow Analysis")
                else:
                    self.analysis_running = True
                    analysis_thread = threading.Thread(target=self.run_analysis)
                    analysis_thread.start()

            elif self.event == "Test threshold":
                self.test_threshold()

            elif self.event == "Test filter":
                self.test_filter()

        # Close the window
        self.window.close()

    def run_analysis(self):
            output = {}     # dictionnary for output
            image_path = self.values["image_path"]
            output_folder = self.values["output_path"].replace("/","\\")
            pixel_size = float(self.values["pixel_size"])
            frame_time = float(self.values["frame_time"])
            filter_size = int(self.values["filter_size"]) if self.values["filter_size"] else None
            threshold = float(self.values["threshold"])
            group_name = self.values["group_name"] if self.values["Custom"] else None

            if self.values["method_hardcore"]:
                thresholding_method = "Hardcore"
            else:
                thresholding_method = "Quantile"

            ind_profile = self.values["individual_profiles"]
            total_profile = self.values["total_profile"]
            csv_table = self.values["csv_table"]
            paths = self.values["image_path"].split(";")
            total_means = []
            labels = []
            if self.values["Filename"]:
                print("args from filename not yet supported!")
            else:
                for path in paths:
                    exp = ky.Kymo(path.replace("/","\\"), pixel_size=pixel_size, frame_time=frame_time)
                    means, se = exp.generate_kymo(threshold=threshold, thresholding_method=thresholding_method, save_profile=ind_profile, filter_size=filter_size, output_folder=output_folder)
                    total_means.append(means)
                    output[exp.name] = [group_name, means]

                if total_profile:
                    # plot total profile (mean of means)
                    
                    mean_velocities = savgol_filter(np.mean(total_means, axis=0),5,2) # compute the mean velocities for every dv position and smooth them
                    se_velocities = savgol_filter(np.std(total_means,axis=0) / np.sqrt(len(total_means)),5,2) # compute the se for every dv position and smooth them
                    
                    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
                    plt.style.use('Solarize_Light2')
                    ax.set_title(group_name+" CSF profile")
                    ax.set_xlabel(r"Dorso-ventral position [$\mu$m]")
                    ax.set_ylabel(r"Average rostro-caudal velocity [$\mu$m/s]")
                    dv_axis = np.arange(-(len(mean_velocities)-(len(mean_velocities)-np.nonzero(mean_velocities)[0][0])),len(mean_velocities)-np.nonzero(mean_velocities)[0][0])*pixel_size # find start of canal based on first non zero speed
                    ax.plot(dv_axis,mean_velocities) 
                    # Plot grey bands for the standard error
                    ax.fill_between(dv_axis, mean_velocities - se_velocities, mean_velocities + se_velocities, color='grey', alpha=0.3, label='Standard Error')
                    ax.legend()
                    if output_folder:
                        fig.savefig(output_folder+"\\"+group_name+"_total_vel_threshold"+str(np.round(exp.threshold,1))+"_filter"+str(filter_size)+'.png')   # save the figure to file
                    else:
                        fig.savefig(group_name+"_total_vel_threshold"+str(np.round(exp.threshold,1))+"_filter"+str(filter_size)+'.png')   # save the figure to file
                    
                    plt.close(fig)    # close the figure window

                if csv_table:
                    # save data as csv
                    output = pd.DataFrame(output)
                    csv_filename = f"{output_folder}\\{group_name}_csf_flow_results.csv"
                    output.to_csv(csv_filename, index=False)

                self.analysis_running = False
                sg.popup("Analysis Completed", title="CSF Flow Analysis")



Gui = GUI()
Gui.start()


