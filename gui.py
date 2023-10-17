import PySimpleGUI as sg
from funcs import kymo as ky
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import threading
import sys
from io import StringIO
import time
import warnings
import shutil 
import subprocess
import os


class GUI:
    def __init__(self):
        warnings.filterwarnings("ignore", category=UserWarning)

        sg.theme("Default1")
        # Define the layout of the GUI
        self.output_element = sg.Multiline(size=(100, 10), key="-OUTPUT-", autoscroll=True) #for console display
        self.layout = [
            [sg.Text("CSF Flow Analysis", font=("Helvetica", 20))],
            [sg.Column([
            [sg.Text("Input(s):         "), sg.InputText(key="image_path"), sg.FilesBrowse()],
            [sg.Text("Output Folder:"), sg.InputText(key="output_path"), sg.FolderBrowse()],
            [sg.TabGroup([
            [sg.Tab("Settings",layout=[
            [sg.Text("Pixel Size (um):"), sg.Combo([0.189,0.21666666666666673,0.16250000000000003],key="pixel_size", size=(6,2), default_value = 0.16250000000000003)],
            [sg.Text("Frame Time (s):"), sg.Combo([0.291,0.1],key="frame_time", size=(6,2), default_value = 0.159)],
            [sg.Text("Filter size (px):"), sg.InputText(key="filter_size", size=(6,2), default_text = None)],
            [sg.Text("Threshold:"), sg.InputText(key="threshold", size=(6,2), default_text = 0.5)],
            [sg.Text("Smoothing window:"), sg.InputText(key="smoothwindow", size=(6,2), default_text = 20),sg.Text("Smoothing polyorder:"), sg.InputText(key="smoothpoly", size=(6,2), default_text = 3)],
            [sg.Text("Thresholding Method:")],
            [sg.Radio("Hardcore", "thresholding", key="method_hardcore"),
            sg.Radio("Quantile", "thresholding", key="method_quantile", default=True)]]
            )],
            [sg.Tab("Output",layout=[
            [sg.Text("Naming Method:")],
            [sg.Radio("Filename", "naming_method", key="Filename")],
            [sg.Radio("Custom", "naming_method", key="Custom", default=True),
            sg.Text("Group name:"), sg.InputText(key="group_name", default_text = "GroupName")],
            [sg.Text("Outputs:")],
            [sg.Checkbox("Individual flow profiles", key="individual_profiles", default=True)],
            [sg.Checkbox("Total flow profile", key="total_profile", default=True)],
            [sg.Checkbox("Profile overlay", key="profile_overlay", default=True)],
            [sg.Checkbox("CSV Data Table", key="csv_table", default=True)],
            ]
            )],
            ])]
            ], element_justification='left')],
            [sg.Column([
            [sg.Button("Test threshold"), sg.Button("Test filter")],
            [sg.Button("Run Analysis"), sg.Button("Exit"), sg.Button("Clear cache"),sg.Button("Test"),sg.Button("Threads")],
            [sg.HorizontalSeparator()],
            [sg.Text("Progress:"),
            sg.ProgressBar(100, orientation="h", size=(50, 20), key="progressbar")],  
            #[sg.Text("Log:")],  
            #[self.output_element]
            ], element_justification='left')],
        ]
     


        # Create the window
        self.window = sg.Window("CSF Flow Analysis GUI", self.layout, element_justification="center")
        self.analysis_running = False

        


    def start(self):
        
        # Event loop
        while True:
            self.event, self.values = self.window.read()
            self.done_test = threading.Event()


            if self.event == sg.WIN_CLOSED or self.event == "Exit":
                # Close the window
                self.window.close()
               
                break

            elif self.event == "Run Analysis":

                if self.analysis_running:
                    sg.popup("Analysis in progress please wait...", title="CSF Flow Analysis")
                else:
                    self.analysis_running = True
                    self.analysis_thread = threading.Thread(target=self.run_analysis)
                    self.analysis_thread.start()
                    

            elif self.event == "Test threshold":
                self.test_threshold()

            elif self.event == "Test filter":
                self.test_filter()
            
            elif self.event == "Clear cache":
                # clear the cache if you modified (ex:rotation) any input images
                shutil.rmtree("cache")
            
            elif self.event == "Test":
                self.test_thread = threading.Thread(target=self.test)
                self.test_thread.start()
            
                
            elif self.event == "Threads":
                print(self.done_test.is_set())
                 # get a list of all active threads
                threads = threading.enumerate()
                # report the name of all active threads
                for thread in threads:
                    print(thread.name)
                
    
    def get_console_output(self,stop_console):
        # Create a buffer to capture console output
        self.output_buffer = StringIO(newline="\n")

        # Redirect standard output to the buffer
        sys.stdout = self.output_buffer

        while not stop_console.is_set():
            # Update the output element with captured console output
            self.output_element.update(self.output_buffer.getvalue())
            time.sleep(0.2)
        sys.stdout = sys.__stdout__
    
        
    def test(self):
       
            stop_console = threading.Event()
            self.console_thread = threading.Thread(target=self.get_console_output,args=(stop_console,))
            self.console_thread.start()

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
            paths = self.values["image_path"].split(";")
            for ind, path in enumerate(paths):
                exp = ky.Kymo(path.replace("/","\\"), pixel_size=pixel_size, frame_time=frame_time)
                exp.generate_kymo(threshold=threshold, thresholding_method=thresholding_method, filter_size=filter_size, output_folder=output_folder,dash=True)
            del exp
            # terminate threads
            stop_console.set()
            #self.console_thread.join()
            del self.console_thread

            stop_console.clear()


    def run_analysis(self):
            
            #stop_console = threading.Event()
            #self.console_thread = threading.Thread(target=self.get_console_output,args=(stop_console,))
            #self.console_thread.start()

            output = {'name': [], 'group': [], 'means': [],'extremum': [], 'minimum': []}     # dictionary for output
            image_path = self.values["image_path"]
            output_folder = self.values["output_path"].replace("/","\\")
            pixel_size = float(self.values["pixel_size"])
            frame_time = float(self.values["frame_time"])
            filter_size = int(self.values["filter_size"]) if self.values["filter_size"] else None
            threshold = float(self.values["threshold"])
            group_name = self.values["group_name"] if self.values["Custom"] else None
            gol_parms = (int(self.values["smoothwindow"]),int(self.values["smoothpoly"]))

            if self.values["method_hardcore"]:
                thresholding_method = "Hardcore"
            else:
                thresholding_method = "Quantile"

            ind_profile = self.values["individual_profiles"]
            total_profile = self.values["total_profile"]
            profile_overlay = self.values["profile_overlay"]
            csv_table = self.values["csv_table"]
            paths = self.values["image_path"].split(";")
            total_means = []
            labels = []
            if self.values["Filename"]:
                print("args from filename not yet supported!")
            else:
                for ind, path in enumerate(paths):
                    exp = ky.Kymo(path.replace("/","\\"), pixel_size=pixel_size, frame_time=frame_time)
                    means, se = exp.generate_kymo(threshold=threshold, thresholding_method=thresholding_method, save_profile=ind_profile, filter_size=filter_size, output_folder=output_folder, gol_parms = gol_parms)
                    total_means.append(means)
                    output["name"].append(exp.name.replace("_cropped","").replace(".tif",""))
                    output["group"].append(group_name)
                    output["means"].append(means.tolist())
                    output["extremum"].append(np.max(means))
                    output["minimum"].append(np.min(means))
                    labels.append(exp.name)
                    del exp
                    self.window["progressbar"].update((ind+1)/len(paths)*100)

                if csv_table:
                    # save data as csv (all in one table)
                    print("Saving csv")
                    df = pd.DataFrame(data=output)
                    
                    csv_filename = f"{output_folder}\\{group_name}_csf_flow_results_t_{threshold}_f_{filter_size}.csv"  # little jim jam to add the threshold to the filename
                    df.to_csv(csv_filename, index=False)

                    # make all the arrays start at the same location
                    for ind,array in enumerate(total_means):
                        total_means[ind] = array[np.nonzero(array)[0]]

                    # pad the arrays if not same size
                    # Find the maximum length of all arrays
                    max_length = max(len(arr) for arr in total_means)

                    # Pad each array to match the maximum length
                    total_means = [np.pad(arr, (5, max_length - len(arr)+5), mode='constant', constant_values=0) for arr in total_means]

                    # save all in single files
                    for name, vels in zip(output['name'],total_means):
                        try:
                            
                            dv_axis = np.arange(-(len(vels)-(len(vels)-np.nonzero(vels)[0][0])),len(vels)-np.nonzero(vels)[0][0])*pixel_size # find start of canal based on first non zero speed
                            df_ind = pd.DataFrame({"x-axis":dv_axis, "mean_vels":vels})
                            outdir =f"{output_folder}\\csv_{group_name}_results_thresh_{threshold}_filt_{filter_size}" 
                            ind_csv_filename = f"{name.replace('.tif','')}.csv"

                            if not os.path.exists(outdir):
                                os.mkdir(outdir)
                            df_ind.to_csv(outdir+"\\"+ind_csv_filename,index=False)
                        except:
                            print("Sorry Not enough traces were detected! Skipping file \nTry different threshold")



                if total_profile or profile_overlay:

                    # plot total profile (mean of means)
                    # make all the arrays start at the same location
                    for ind,array in enumerate(total_means):
                        total_means[ind] = array[np.nonzero(array)[0]]

                    # pad the arrays if not same size
                    # Find the maximum length of all arrays
                    max_length = max(len(arr) for arr in total_means)

                    # Pad each array to match the maximum length
                    total_means = [np.pad(arr, (5, max_length - len(arr)+5), mode='constant', constant_values=0) for arr in total_means]

                    if total_profile:
                        # get mean velocities and se
                        mean_velocities = savgol_filter(np.mean(total_means, axis=0),5,2) # compute the mean velocities for every dv position and smooth them
                        se_velocities = savgol_filter(np.std(total_means,axis=0) / np.sqrt(len(total_means)),5,2) # compute the se for every dv position and smooth them
                        
                        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create 1 figure & 1 axis
                        ax.set_title(group_name+" CSF profile")
                        ax.set_xlabel(r"Absolute Dorso-ventral position [$\mu$m]")
                        ax.set_ylabel(r"Average rostro-caudal velocity [$\mu$m/s]")
                        dv_axis = np.arange(-(len(mean_velocities)-(len(mean_velocities)-np.nonzero(mean_velocities)[0][0])),len(mean_velocities)-np.nonzero(mean_velocities)[0][0])*pixel_size # find start of canal based on first non zero speed
                        ax.plot(dv_axis,mean_velocities) 
                        # Plot grey bands for the standard error
                        ax.fill_between(dv_axis, mean_velocities - se_velocities, mean_velocities + se_velocities, color='grey', alpha=0.3, label='Standard Error')
                        ax.legend()

                        if output_folder:
                            fig.savefig(output_folder+"\\"+group_name+"_total_vel_t"+str(np.round(threshold,1))+"_f"+str(filter_size)+'.png')   # save the figure to file
                        else:
                            fig.savefig(group_name+"_total_vel_t"+str(np.round(threshold,1))+"_f"+str(filter_size)+'.png')   # save the figure to file
                        
                        plt.close(fig)    # close the figure window

                    if profile_overlay:

                        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create 1 figure & 1 axis
                        ax.set_title(group_name+" CSF profile")
                        ax.set_xlabel(r"Absolute Dorso-ventral position [$\mu$m]")
                        ax.set_ylabel(r"Average rostro-caudal velocity [$\mu$m/s]")
                        dv_axis = np.arange(-(len(total_means[0])-(len(total_means[0])-np.nonzero(total_means[0])[0][0])),len(total_means[0])-np.nonzero(total_means[0])[0][0])*pixel_size # find start of canal based on first non zero speed
                        for profile,nom in zip(total_means,output["name"]):
                            ax.plot(dv_axis,profile,alpha=0.6,label=nom) 
                            ax.fill_between(dv_axis,profile, 0, alpha=0.1)
                        ax.legend()

                        if output_folder:
                            fig.savefig(output_folder+"\\"+group_name+"_profile_overlay_t"+str(np.round(threshold,1))+"_f"+str(filter_size)+'.png')   # save the figure to file
                        else:
                            fig.savefig(group_name+"_profile_overlay_t"+str(np.round(threshold,1))+"_f"+str(filter_size)+'.png')   # save the figure to file
                        
                        plt.close(fig)    # close the figure window
                # show where the results are outputed
                subprocess.Popen(f'explorer "{output_folder}"')

                self.analysis_running = False
                #stop_console.set()
                #self.console_thread.join()
                #del self.console_thread
                #stop_console.clear()
                self.window["progressbar"].update(0)
                

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


Gui = GUI()
Gui.start()


