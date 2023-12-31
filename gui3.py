import eel

import warnings
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


#warnings.simplefilter(action='ignore', category=FutureWarning)

eel.init('funcs/gui')

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


@eel.expose 
def select_input():
    print("Prompting user for input...")
    paths = sg.popup_get_file("", no_window=True, default_extension=".tif", multiple_files=True)
    print(paths)
    paths = [path+"<br>" for path in paths] # FOR DISPLAY IN HTML
    return paths

@eel.expose 
def select_output():
    path = sg.popup_get_folder("", no_window=True)
    print(path)
    return path

@eel.expose 
def test(image_path,output_folder,pixel_size,frame_time,filter_size,threshold,ind_profile,total_profile,csv_table):
        
        output_folder = output_folder.replace("/","\\")
        filter_size = filter_size if filter_size else None
        paths = image_path.split(";")

        total_means = []
        labels = []
        for ind, path in enumerate(paths):
            exp = ky.Kymo(path.replace("/","\\"), pixel_size=pixel_size, frame_time=frame_time)
            exp.generate_kymo(threshold=threshold, thresholding_method="Quantile", filter_size=filter_size, output_folder=output_folder)
        del exp

@eel.expose
def clear_cache():
    shutil.rmtree("cache")

@eel.expose 
def run_analysis(self):
        
        stop_console = threading.Event()
        self.console_thread = threading.Thread(target=self.get_console_output,args=(stop_console,))
        self.console_thread.start()

        output = {'name': [], 'group': [], 'means': []}     # dictionnary for output
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
            for ind, path in enumerate(paths):
                exp = ky.Kymo(path.replace("/","\\"), pixel_size=pixel_size, frame_time=frame_time)
                means, se = exp.generate_kymo(threshold=threshold, thresholding_method=thresholding_method, save_profile=ind_profile, filter_size=filter_size, output_folder=output_folder)
                total_means.append(means)
                output["name"].append(exp.name)
                output["group"].append(group_name)
                output["means"].append(means[0])
                labels.append(exp.name)
                del exp
                self.window["progressbar"].update((ind+1)/len(paths)*100)

            if csv_table:
                # save data as csv
                print("Saving csv")
                df = pd.DataFrame(data=output)
                print(df)
                csv_filename = f"{output_folder}\\{group_name}_csf_flow_results.csv"
                df.to_csv(csv_filename, index=False)

            if total_profile:

                # plot total profile (mean of means)
                # make all the arrays start at the same location
                for ind,array in enumerate(total_means):
                    total_means[ind] = array[np.nonzero(array)[0]]

                # pad the arrays if not same size
                # Find the maximum length of all arrays
                max_length = max(len(arr) for arr in total_means)

                # Pad each array to match the maximum length
                total_means = [np.pad(arr, (5, max_length - len(arr)+5), mode='constant', constant_values=0) for arr in total_means]

                # get mean velocities and se
                mean_velocities = savgol_filter(np.mean(total_means, axis=0),5,2) # compute the mean velocities for every dv position and smooth them
                se_velocities = savgol_filter(np.std(total_means,axis=0) / np.sqrt(len(total_means)),5,2) # compute the se for every dv position and smooth them
                
                fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
                plt.style.use('Solarize_Light2')
                ax.set_title(group_name+" CSF profile")
                ax.set_xlabel(r"Absolute Dorso-ventral position [$\mu$m]")
                ax.set_ylabel(r"Average rostro-caudal velocity [$\mu$m/s]")
                dv_axis = np.arange(-(len(mean_velocities)-(len(mean_velocities)-np.nonzero(mean_velocities)[0][0])),len(mean_velocities)-np.nonzero(mean_velocities)[0][0])*pixel_size # find start of canal based on first non zero speed
                ax.plot(dv_axis,mean_velocities) 
                # Plot grey bands for the standard error
                ax.fill_between(dv_axis, mean_velocities - se_velocities, mean_velocities + se_velocities, color='grey', alpha=0.3, label='Standard Error')
                ax.legend()

                if output_folder:
                    fig.savefig(output_folder+"\\"+group_name+"_total_vel_threshold"+str(np.round(threshold,1))+"_filter"+str(filter_size)+'.png')   # save the figure to file
                else:
                    fig.savefig(group_name+"_total_vel_threshold"+str(np.round(threshold,1))+"_filter"+str(filter_size)+'.png')   # save the figure to file
                
                plt.close(fig)    # close the figure window

                # scatter plot
                """
                fig2, ax2 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
                plt.style.use('Solarize_Light2')
                ax2.set_title(group_name+" CSF profile")
                ax2.set_xlabel(r"Absolute Dorso-ventral position [$\mu$m]")
                ax2.set_ylabel(r"Average rostro-caudal velocity [$\mu$m/s]")
                
                y = df["means"].values.tolist()
                print(y)
                # make all the arrays start at the same location
                for ind,array in enumerate(y):
                    y[ind] = array[np.nonzero(array)[0]]
                # Pad each array to match the maximum length
                y = [np.pad(arr, (5, max_length - len(arr)+5), mode='constant', constant_values=0) for arr in y]
                x =  np.arange(-(len(y)-(len(y)-np.nonzero(y)[0][0])),len(y)-np.nonzero(y)[0][0])*pixel_size # find start of canal based on first non zero speed
                ax2.scatter(x,y,df["name"].values.tolist()) 
                ax2.legend()

                if output_folder:
                    fig2.savefig(output_folder+"\\"+group_name+"_scatter_total_vel_threshold"+str(np.round(threshold,1))+"_filter"+str(filter_size)+'.png')   # save the figure to file
                else:
                    fig2.savefig(group_name+"_scatter_total_vel_threshold"+str(np.round(threshold,1))+"_filter"+str(filter_size)+'.png')   # save the figure to file
                
                plt.close(fig2)    # close the figure window
                """

            # show where the results are outputted
            subprocess.Popen(f'explorer "{output_folder}"')

            self.analysis_running = False
            stop_console.set()
            self.console_thread.join()
            del self.console_thread
            stop_console.clear()
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






eel.start('index.html')