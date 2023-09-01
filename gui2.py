
# test for another GUI

import tkinter as tk
from tkinter import filedialog

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
from funcs import kymo as ky
from tkinter import ttk
from ttkthemes import ThemedStyle


class GUI:
    def __init__(self, root):

        warnings.filterwarnings("ignore", category=UserWarning)

        self.root = root
        self.root.title("CSF Flow Analysis")
        self.root.geometry("800x600")
        self.analysis_running = False

        self.style = ThemedStyle(root)
        self.style.set_theme("adapta")

        self.output_text = tk.Text(root, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self.create_widgets()

        self.output_buffer = StringIO(newline="\n")
        sys.stdout = self.output_buffer

    def create_widgets(self):
        # Create a notebook to organize tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        settings_tab = ttk.Frame(notebook)
        output_tab = ttk.Frame(notebook)

        notebook.add(settings_tab, text="Settings")
        notebook.add(output_tab, text="Log")

        # Settings Tab
        settings_frame = ttk.LabelFrame(settings_tab)
        settings_frame.pack(padx=10, pady=10)

        self.image_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.pixel_size_var = tk.StringVar(value="0.189")
        self.frame_time_var = tk.StringVar(value="0.1")
        self.filter_size_var = tk.StringVar()
        self.threshold_var = tk.StringVar(value="0.5")
        self.method_var = tk.StringVar(value="Quantile")
        self.naming_method_var = tk.StringVar(value="Custom")
        self.group_name_var = tk.StringVar(value="GroupName")
        self.individual_profiles_var = tk.BooleanVar(value=True)
        self.total_profile_var = tk.BooleanVar(value=True)
        self.csv_table_var = tk.BooleanVar(value=True)

        input_label = ttk.Label(settings_frame, text="Input(s):")
        input_label.grid(row=0, column=0, sticky=tk.W)

        input_entry = ttk.Entry(settings_frame, textvariable=self.image_path_var)
        input_entry.grid(row=0, column=1, padx=5)

        input_browse_button = ttk.Button(settings_frame, text="Browse", command=self.browse_input)
        input_browse_button.grid(row=0, column=2, padx=5)

        output_label = ttk.Label(settings_frame, text="Output Folder:")
        output_label.grid(row=1, column=0, sticky=tk.W)

        output_entry = ttk.Entry(settings_frame, textvariable=self.output_path_var)
        output_entry.grid(row=1, column=1, padx=5)

        output_browse_button = ttk.Button(settings_frame, text="Browse", command=self.browse_output)
        output_browse_button.grid(row=1, column=2, padx=5)

        pixel_size_label = ttk.Label(settings_frame, text="Pixel Size (um):")
        pixel_size_label.grid(row=2, column=0, sticky=tk.W)

        pixel_size_entry = ttk.Entry(settings_frame, textvariable=self.pixel_size_var)
        pixel_size_entry.grid(row=2, column=1, padx=5)

        frame_time_label = ttk.Label(settings_frame, text="Frame Time (s):")
        frame_time_label.grid(row=3, column=0, sticky=tk.W)

        frame_time_entry = ttk.Entry(settings_frame, textvariable=self.frame_time_var)
        frame_time_entry.grid(row=3, column=1, padx=5)

        filter_size_label = ttk.Label(settings_frame, text="Filter size (px):")
        filter_size_label.grid(row=4, column=0, sticky=tk.W)

        filter_size_entry = ttk.Entry(settings_frame, textvariable=self.filter_size_var)
        filter_size_entry.grid(row=4, column=1, padx=5)

        threshold_label = ttk.Label(settings_frame, text="Threshold:")
        threshold_label.grid(row=5, column=0, sticky=tk.W)

        threshold_entry = ttk.Entry(settings_frame, textvariable=self.threshold_var)
        threshold_entry.grid(row=5, column=1, padx=5)

        threshold_method_label = ttk.Label(settings_frame, text="Thresholding Method:")
        threshold_method_label.grid(row=6, column=0, sticky=tk.W)

        hardcore_radio = ttk.Radiobutton(settings_frame, text="Hardcore", variable=self.method_var, value="Hardcore")
        hardcore_radio.grid(row=6, column=1, padx=5, sticky=tk.W)

        quantile_radio = ttk.Radiobutton(settings_frame, text="Quantile", variable=self.method_var, value="Quantile")
        quantile_radio.grid(row=6, column=2, padx=5, sticky=tk.W)

        output_naming_label = ttk.Label(settings_frame, text="Naming Method:")
        output_naming_label.grid(row=7, column=0, sticky=tk.W)

        filename_radio = ttk.Radiobutton(settings_frame, text="Filename", variable=self.naming_method_var, value="Filename")
        filename_radio.grid(row=7, column=1, padx=5, sticky=tk.W)

        custom_radio = ttk.Radiobutton(settings_frame, text="Custom", variable=self.naming_method_var, value="Custom")
        custom_radio.grid(row=7, column=2, padx=5, sticky=tk.W)

        group_name_label = ttk.Label(settings_frame, text="Group name:")
        group_name_label.grid(row=8, column=1, padx=5, sticky=tk.W)

        group_name_entry = ttk.Entry(settings_frame, textvariable=self.group_name_var)
        group_name_entry.grid(row=8, column=2, padx=5, sticky=tk.W)

        outputs_label = ttk.Label(settings_frame, text="Outputs:")
        outputs_label.grid(row=9, column=0, sticky=tk.W)

        individual_profiles_check = ttk.Checkbutton(settings_frame, text="Individual flow profiles", variable=self.individual_profiles_var)
        individual_profiles_check.grid(row=10, column=0, sticky=tk.W)

        total_profile_check = ttk.Checkbutton(settings_frame, text="Total flow profile", variable=self.total_profile_var)
        total_profile_check.grid(row=11, column=0, sticky=tk.W)

        csv_table_check = ttk.Checkbutton(settings_frame, text="CSV Data Table", variable=self.csv_table_var)
        csv_table_check.grid(row=12, column=0, sticky=tk.W)

        buttons_frame = ttk.Frame(settings_tab)
        buttons_frame.pack(pady=10)

        test_threshold_button = ttk.Button(buttons_frame, text="Test threshold", command=self.test_threshold)
        test_threshold_button.grid(row=0, column=0, padx=5)

        test_filter_button = ttk.Button(buttons_frame, text="Test filter", command=self.test_filter)
        test_filter_button.grid(row=0, column=1, padx=5)

        run_analysis_button = ttk.Button(buttons_frame, text="Run Analysis", command=self.run_analysis)
        run_analysis_button.grid(row=0, column=2, padx=5)

        clear_cache_button = ttk.Button(buttons_frame, text="Clear cache", command=self.clear_cache)
        clear_cache_button.grid(row=0, column=3, padx=5)

        # Output Tab
        output_text_frame = ttk.LabelFrame(output_tab, text="Output Console")
        output_text_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(output_text_frame, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def browse_input(self):
        paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.tif")])
        if paths:
            self.image_path_var.set(";".join(paths))

    def browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_path_var.set(folder)

    def run_analysis(self):
        if self.analysis_running:
            self.log("Analysis in progress please wait...")
        else:
            self.analysis_running = True
            analysis_thread = threading.Thread(target=self.run_analysis_thread)
            analysis_thread.start()

    def run_analysis_thread(self):
        stop_console = threading.Event()
        console_thread = threading.Thread(target=self.get_console_output, args=(stop_console,))
        console_thread.start()

        output = {'name': [], 'group': [], 'means': []}
        image_paths = self.image_path_var.get().split(";")
        output_folder = self.output_path_var.get()
        pixel_size = float(self.pixel_size_var.get())
        frame_time = float(self.frame_time_var.get())
        filter_size = int(self.filter_size_var.get()) if self.filter_size_var.get() else None
        threshold = float(self.threshold_var.get())
        group_name = self.group_name_var.get() if self.naming_method_var.get() == "Custom" else None

        if self.method_var.get() == "Hardcore":
            thresholding_method = "Hardcore"
        else:
            thresholding_method = "Quantile"

        ind_profile = self.individual_profiles_var.get()
        total_profile = self.total_profile_var.get()
        csv_table = self.csv_table_var.get()
        total_means = []

        for ind, path in enumerate(image_paths):
            exp = ky.Kymo(path.replace("/", "\\"), pixel_size=pixel_size, frame_time=frame_time)
            means, se = exp.generate_kymo(threshold=threshold, thresholding_method=thresholding_method,
                                          save_profile=ind_profile, filter_size=filter_size, output_folder=output_folder)
            total_means.append(means)
            output["name"].append(exp.name)
            output["group"].append(group_name)
            output["means"].append(means[0])
            self.log(f"Processed image {ind + 1}/{len(image_paths)}: {exp.name}")
            del exp

        if csv_table:
            df = pd.DataFrame(data=output)
            csv_filename = f"{output_folder}/{group_name}_csf_flow_results.csv"
            df.to_csv(csv_filename, index=False)
            self.log(f"Saved data as CSV: {csv_filename}")

        if total_profile:
            max_length = max(len(arr) for arr in total_means)
            total_means = [np.pad(arr, (5, max_length - len(arr) + 5), mode='constant', constant_values=0) for arr in
                           total_means]
            mean_velocities = savgol_filter(np.mean(total_means, axis=0), 5, 2)
            se_velocities = savgol_filter(np.std(total_means, axis=0) / np.sqrt(len(total_means)), 5, 2)

            fig, ax = plt.subplots(nrows=1, ncols=1)
            plt.style.use('Solarize_Light2')
            ax.set_title(f"{group_name} CSF profile")
            ax.set_xlabel(r"Absolute Dorso-ventral position [$\mu$m]")
            ax.set_ylabel(r"Average rostro-caudal velocity [$\mu$m/s]")
            dv_axis = np.arange(-(len(mean_velocities) - (len(mean_velocities) - np.nonzero(mean_velocities)[0][0])),
                               len(mean_velocities) - np.nonzero(mean_velocities)[0][0]) * pixel_size
            ax.plot(dv_axis, mean_velocities)
            ax.fill_between(dv_axis, mean_velocities - se_velocities, mean_velocities + se_velocities, color='grey',
                            alpha=0.3, label='Standard Error')
            ax.legend()

            if output_folder:
                fig.savefig(f"{output_folder}/{group_name}_total_vel_threshold{threshold:.1f}_filter{filter_size}.png")
            else:
                fig.savefig(f"{group_name}_total_vel_threshold{threshold:.1f}_filter{filter_size}.png")
            plt.close(fig)

            subprocess.Popen(f'explorer "{output_folder}"')

        self.analysis_running = False
        stop_console.set()
        sys.stdout = sys.__stdout__
        self.log("Analysis completed.")

    def clear_cache(self):
        shutil.rmtree("cache")
        self.log("Cache cleared.")

    def test_threshold(self):
        pixel_size = float(self.pixel_size_var.get())
        frame_time = float(self.frame_time_var.get())
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.tif")])
        if path:
            exp = ky.Kymo(path.replace("/", "\\"), pixel_size=pixel_size, frame_time=frame_time)
            exp.test_threshold()
            self.log("Threshold test completed.")

    def test_filter(self):
        pixel_size = float(self.pixel_size_var.get())
        frame_time = float(self.frame_time_var.get())
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.tif")])
        if path:
            exp = ky.Kymo(path.replace("/", "\\"), pixel_size=pixel_size, frame_time=frame_time)
            exp.test_filter()
            self.log("Filter test completed.")

    def get_console_output(self, stop_console):
        while not stop_console.is_set():
            self.output_text.delete(1.0, ttk.END)
            self.output_text.insert(ttk.END, self.output_buffer.getvalue())
            self.root.update()
            time.sleep(0.2)

    def log(self, message):
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S] ")
        self.output_text.insert(ttk.END, timestamp + message + "\n")
        self.root.update()
        self.output_buffer.write(timestamp + message + "\n")

if __name__ == "__main__":
    root = tk.Tk()
    
    app = GUI(root)
    root.mainloop()
