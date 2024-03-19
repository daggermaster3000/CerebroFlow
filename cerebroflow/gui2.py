import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLineEdit, QFileDialog, QLabel, QRadioButton, QComboBox, QCheckBox, QProgressBar, QTabWidget, QGridLayout, QSizePolicy,QDoubleSpinBox, QHBoxLayout, QMessageBox
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QFont,QPixmap
import os
import cerebroflow.kymo as ky
import subprocess
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import dominate
from dominate import document
from dominate.tags import *
from datetime import datetime


class GUI(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        tabWidget = QTabWidget()

        # Tab 1: Input
        inputTab = QWidget()
        inputLayout = QGridLayout(inputTab)

        # Input images
        self.imagePathInput = QLineEdit(self)
        self.imagePathInput.setPlaceholderText("Enter image path")
        inputLayout.addWidget(QLabel("Input(s):", self), 0, 0)
        inputLayout.addWidget(self.imagePathInput, 1, 0)
        browseButton = QPushButton('Browse', self)
        browseButton.clicked.connect(self.browseImage)
        inputLayout.addWidget(browseButton, 1, 1)

        # Output folder
        self.outputPathInput = QLineEdit(self)
        self.outputPathInput.setPlaceholderText("Enter output folder path")
        inputLayout.addWidget(QLabel("Output Folder:", self), 2, 0)
        inputLayout.addWidget(self.outputPathInput, 3, 0)
        folderBrowseButton = QPushButton('Browse', self)
        folderBrowseButton.clicked.connect(self.browseOutputFolder)
        inputLayout.addWidget(folderBrowseButton, 3, 1)
        inputLayout.addWidget(QLabel("", self))
        inputLayout.addWidget(QLabel("", self))
        inputLayout.addWidget(QLabel("", self))

        # Tab 2: Settings
        settingsTab = QWidget()
        settingsLayout = QGridLayout(settingsTab)
        settingsLayout.addWidget(QLabel("General", self), 0, 0, 1, 2)

        settingsLayout.addWidget(QLabel("Pixel Size (um):", self), 1, 0)
        self.pixelSizeCombo = QComboBox(self)
        self.pixelSizeCombo.setEditable(True)
        self.pixelSizeCombo.addItems(["0.189", "0.21666666666666673", "0.16250000000000003"])
        settingsLayout.addWidget(self.pixelSizeCombo, 1, 1)
        self.useMeta = QCheckBox('Use metadata', self, checked=True)
        settingsLayout.addWidget(self.useMeta, 1, 2)

        settingsLayout.addWidget(QLabel("Frame Time (s):", self), 2, 0)
        self.frameTimeCombo = QComboBox(self)
        self.frameTimeCombo.setEditable(True)
        self.frameTimeCombo.addItems(["0.291", "0.1"])
        settingsLayout.addWidget(self.frameTimeCombo, 2, 1)

        self.filterSizeInput = QDoubleSpinBox(self)
        settingsLayout.addWidget(QLabel("Filter size (px):", self), 3, 0)
        settingsLayout.addWidget(self.filterSizeInput, 3, 1)

        settingsLayout.addWidget(QLabel("Threshold:", self), 4, 0)
        self.Threshold = QDoubleSpinBox(self)
        self.Threshold.setValue(0.5)
        self.Threshold.setSingleStep(0.1)
        settingsLayout.addWidget(self.Threshold, 4, 1)

        settingsLayout.addWidget(QLabel("Smoothing", self), 5, 0)
        settingsLayout.addWidget(QLabel("Smoothing window:", self), 6, 0)
        self.smoothWinSpinBox = QDoubleSpinBox(self)
        self.smoothWinSpinBox.setValue(30)
        settingsLayout.addWidget(self.smoothWinSpinBox, 6, 1)

        settingsLayout.addWidget(QLabel("Smoothing polyorder:", self), 7, 0)
        self.smoothPolySpinBox = QDoubleSpinBox(self)
        self.smoothPolySpinBox.setValue(3)
        settingsLayout.addWidget(self.smoothPolySpinBox, 7, 1)

        self.golayRadio = QRadioButton('Golay', self)
        settingsLayout.addWidget(self.golayRadio, 8, 0)
        self.smoothRadio = QRadioButton('Smooth', self)
        settingsLayout.addWidget(self.smoothRadio, 8, 1)
        self.combineRadio = QRadioButton('Combine', self, checked=True)
        settingsLayout.addWidget(self.combineRadio, 8, 2)

        settingsLayout.addWidget(QLabel("Other", self), 9, 0)
        settingsLayout.addWidget(QLabel("Thresholding Method:", self), 10, 0)
        self.hardcoreRadio = QRadioButton('Hardcore', self)
        settingsLayout.addWidget(self.hardcoreRadio, 11, 0)
        self.quantileRadio = QRadioButton('Quantile', self, checked=True)
        settingsLayout.addWidget(self.quantileRadio, 11, 1)

        # D-V start threshold
        settingsLayout.addWidget(QLabel("D-V start threshold:", self), 12, 0)
        default_dv_thresh = 0.2
        self.dvThreshSpinBox = QDoubleSpinBox(self)
        self.dvThreshSpinBox.setValue(default_dv_thresh)
        self.dvThreshSpinBox.setSingleStep(0.1)
        settingsLayout.addWidget(self.dvThreshSpinBox, 12, 1)

       

        # Tab 3: Output
        outputTab = QWidget()
        outputLayout = QGridLayout(outputTab)

        # Naming Method
        outputLayout.addWidget(QLabel("Naming:", self), 0, 0)
        #filenameRadio = QRadioButton('Filename', self)
        #customRadio = QRadioButton('Custom', self)
        #customRadio.setChecked(True)
        #outputLayout.addWidget(filenameRadio, 1, 0)
        #outputLayout.addWidget(customRadio, 1, 1)

        # Group name
        outputLayout.addWidget(QLabel("Group name:", self), 2, 0)
        self.groupNameInput = QLineEdit(self)
        self.groupNameInput.setPlaceholderText("Enter group name")
        outputLayout.addWidget(self.groupNameInput, 2, 1)

       
        # Outputs
        outputLayout.addWidget(QLabel("Outputs:", self), 4, 0)
        self.individualProfilesCheckbox = QCheckBox('Individual flow profiles', self,checked=True)
        self.totalProfileCheckbox = QCheckBox('Total flow profile', self,checked=True)
        self.profileOverlayCheckbox = QCheckBox('Profile overlay', self,checked=True)
        self.csvTableCheckbox = QCheckBox('CSV Data Table', self,checked=True)
        outputLayout.addWidget(self.individualProfilesCheckbox, 5, 0)
        outputLayout.addWidget(self.totalProfileCheckbox, 6, 0)
        outputLayout.addWidget(self.profileOverlayCheckbox, 7, 0)
        outputLayout.addWidget(self.csvTableCheckbox, 8, 0)

        boldFont = QFont()
        boldFont.setBold(True)
        boldFont.setPointSize(30)

        # some labels bold
        labels_to_make_bold = [outputLayout.itemAtPosition(row, 0).widget() for row in [0, 4]]
        for label in labels_to_make_bold:
            if isinstance(label, QLabel):
                label.setFont(boldFont)

        labels_to_make_bold = [inputLayout.itemAtPosition(row, 0).widget() for row in [0,2]]
        for label in labels_to_make_bold:
            if isinstance(label, QLabel):
                label.setFont(boldFont)

        labels_to_make_bold = [settingsLayout.itemAtPosition(row, 0).widget() for row in [0,5,9]]
        for label in labels_to_make_bold:
            if isinstance(label, QLabel):
                label.setFont(boldFont)


        # Add the tabs
        tabWidget.addTab(inputTab, "Input")
        tabWidget.addTab(settingsTab, "Settings")
        tabWidget.addTab(outputTab, "Output")

        layout.addWidget(tabWidget)

        # Buttons and Progress Bar
        testbuttonLayout = QHBoxLayout()
        test_profile_button = QPushButton('Test Profile', self)
        test_profile_button.clicked.connect(self.testProfile) 
        testbuttonLayout.addWidget(test_profile_button)
        testbuttonLayout.addWidget(QPushButton('Test Threshold', self))
        testbuttonLayout.addWidget(QPushButton('Test Filter', self))
        testbuttonLayout.addWidget(QPushButton('Clear Cache', self))

        view_graphs_button = QPushButton('View Graphs',self)
        #view_graphs_button.clicked.connect(self.start_graph_viewer)
        testbuttonLayout.addWidget(view_graphs_button)
        layout.addLayout(testbuttonLayout)

        buttonLayout = QVBoxLayout()
        run_analysis_button = QPushButton('Run Analysis', self)
        buttonLayout.addWidget(run_analysis_button)
        run_analysis_button.clicked.connect(self.startAnalysisthread) 
        buttonLayout.addWidget(QLabel("", self))
        buttonLayout.addWidget(QLabel("Progress:", self))
        self.Pbar = QProgressBar(self)
        buttonLayout.addWidget(self.Pbar)
        layout.addLayout(buttonLayout)

        self.setLayout(layout)

        self.setGeometry(700, 200, 700, 300)
        self.setWindowTitle('Cerebroflow')
        self.show()


    def browseImage(self):
        options = QFileDialog.Options()
        fileDialog = QFileDialog()
        filePaths, _ = fileDialog.getOpenFileNames(self, "Select Images", "", "All Files (*);;Image Files (*.png *.jpg *.tif)", options=options)
        if filePaths:
            # Display selected files in the input field
            self.imagePathInput.setText(";".join(filePaths))

    def browseOutputFolder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        folderPath = QFileDialog.getExistingDirectory(self, "Select Output Folder", options=options)
        if folderPath:
            self.outputPathInput.setText(folderPath)
        

    def show_popup(self,message):
        # Create a QMessageBox
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Warning")
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        self.popup_showed = True
        # Show the messagebox and handle the result
        result = msg_box.exec_()

    def analysisThreadFinished(self):
        
        print("Analysis is done")  
    
    def generate_overlay_plot(self,root_folder):
        plt.figure(figsize=(10, 6))  # Set figure size as needed

        # Iterate over each folder in the root directory
        for folder_name in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue  # Skip if not a directory
            fig, axs = plt.subplots()
            # Iterate over each CSV file in the current folder
            for csv_file in os.listdir(folder_path):
                if not csv_file.endswith('.csv'):
                    continue  # Skip if not a CSV file
                if csv_file.startswith("._"):
                    continue

                csv_path = os.path.join(folder_path, csv_file)

                # Read the CSV file and overlay the data on the plot

                df = pd.read_csv(csv_path)
                try:
                    plt.plot(df['x-axis'], df['mean_vels'], label=f"{csv_file[:-4]}")
                except:
                    print("Failed to plot")

            ax = plt.gca()
            # Change the x-axis line weight
            ax.spines['bottom'].set_linewidth(2)  

            # Change the y-axis line weight
            ax.spines['left'].set_linewidth(2)  

            axs.spines[['right', 'top']].set_visible(False)

            # Change the x-axis label and tick label font weight
            ax.set_xlabel(r'Dorso-Ventral position $[\mu m]$', weight='bold')


            # Change the y-axis label and tick label font weight
            ax.set_ylabel(r'Rostro-Caudal Velocity $[\mu m/s]$', weight='bold')

            # Set the legend text to bold
            legend = ax.legend()
            for text in legend.get_texts():
                text.set_fontweight('bold')
            #plt.title(f"Overlay Plot for All CSV Files")
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(folder_path,'plots', 'overlay_plot.png'))
            plt.clf()
            plt.close()

    def generate_individual_plots(self, root_folder):
        
        for folder_name in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue  # Skip if not a directory
            

            # Create a sub-subfolder 'plots' for each folder
            folder_plots_path = os.path.join(folder_path, 'plots')
            os.makedirs(folder_plots_path, exist_ok=True)

            # Iterate over each CSV file in the current folder
            for csv_file in os.listdir(folder_path):
                #print(folder_path)
                if not csv_file.endswith('.csv'):
                    continue  # Skip if not a CSV file
                if csv_file.startswith("._"):
                    continue

                csv_path = os.path.join(folder_path, csv_file)
                png_file = os.path.splitext(csv_file)[0] + '.png'
                png_path = os.path.join(folder_plots_path, png_file)

                # Check if the corresponding PNG file already exists
                if os.path.exists(png_path):
                    print(f"Skipping {csv_file}: PNG file already exists")
                    continue

                # Read the CSV file and generate the plot
                try:
                    df = pd.read_csv(csv_path)
                    # plot settings
                    fig, axs = plt.subplots()
                    plt.plot(df['x-axis'], df['mean_vels'],alpha=0.6) 
                    plt.title(f"{csv_file.rstrip('.csv')}")

                    ax = plt.gca()
                    # Change the x-axis line weight
                    ax.spines['bottom'].set_linewidth(2)  

                    # Change the y-axis line weight
                    ax.spines['left'].set_linewidth(2)  

                    axs.spines[['right', 'top']].set_visible(False)

                    # Change the x-axis label and tick label font weight
                    ax.set_xlabel('Absolute Dorso-Ventral position [A.u.]', weight='bold')


                    # Change the y-axis label and tick label font weight
                    ax.set_ylabel('Absolute Rostro-Caudal Velocity [A.u.]', weight='bold')
                    
                    plt.xlabel("Absolute Dorso-Ventral position [um]")
                    plt.ylabel("Rostro-Caudal Velocity [um/s]")
                    plt.savefig(png_path)
                    plt.clf()
                    plt.close()  # Close the plot to release resources
                    print(f"Plot generated for {csv_file}")
                except:
                    print(f"Failed to generate plot for {csv_file}")
                
                

    def createPlots(self):
        output_folder = os.path.normpath(self.outputPathInput.text())
        plt.rcParams['figure.dpi'] = 600
        self.generate_individual_plots(output_folder)
        self.generate_overlay_plot(output_folder)

        print("All plots generated successfully.")
        # check the os to define how we open the folder once the data is processed
        if sys.platform == 'darwin':
            def openFolder(path):
                subprocess.call(['open', '--', path])
        elif sys.platform == 'linux2':
            def openFolder(path):
                subprocess.call(['xdg-open', '--', path])
        elif sys.platform == 'win32':
            def openFolder(path):
                subprocess.call(['explorer', path])
        openFolder(output_folder)


    def updateProgressBar(self, current, total):
        self.Pbar.setValue(int((current / total) * 100))


    def testProfile(self):
            print("Testing...")
            # Get values from the GUI
            image_path = self.imagePathInput.text()
            output_folder = os.path.normpath(self.outputPathInput.text())
            pixel_size = float(self.pixelSizeCombo.currentText())
            frame_time = float(self.frameTimeCombo.currentText())
            filter_size = int(self.filterSizeInput.text()) if self.filterSizeInput.text() else None
            threshold = float(self.Threshold.value())
            group_name = self.groupNameInput.text() if self.groupNameInput.text() else None
            thresholding_method = "Hardcore" if self.hardcoreRadio.isChecked() else "Quantile"
            paths = image_path.split(";")

            for ind, path in enumerate(paths):
                if os.path.isfile(os.path.normpath(path)):
                    exp = ky.Kymo(os.path.normpath(path), pixel_size=pixel_size, frame_time=frame_time, dv_thresh=threshold)
                    exp.generate_kymo(threshold=threshold, thresholding_method=thresholding_method, filter_size=filter_size, output_folder=output_folder,dash=True)
                    del exp
                else:
                    self.show_popup(f"Input Error: {path} is not valid")
        
    def startAnalysisthread(self):
        # Create and start the worker thread
        print("Starting analysis...")
        # Get values from the GUI and store them in a dictionary
        self.popup_showed = False
        self.gui_parms = {
            "image_path": self.imagePathInput.text() if self.imagePathInput.text() else self.show_popup("Please provide an input path"),
            "output_folder": os.path.normpath(self.outputPathInput.text()) if self.outputPathInput.text() else self.show_popup("Please provide an output path"),
            "pixel_size": float(self.pixelSizeCombo.currentText()),
            "frame_time": float(self.frameTimeCombo.currentText()),
            "filter_size": int(self.filterSizeInput.value()) if int(self.filterSizeInput.value()) != 0 else None,
            "threshold": float(self.Threshold.value()),
            "group_name": self.groupNameInput.text() if self.groupNameInput.text() else self.show_popup("Please provide a group name"),
            "thresholding_method": "Hardcore" if self.hardcoreRadio.isChecked() else "Quantile",
            "paths": self.imagePathInput.text().split(";"),
            "gol_parms": (int(self.smoothWinSpinBox.value()), int(self.smoothPolySpinBox.value())),
            "dv_thresh": self.dvThreshSpinBox.value(),
            "filtering_method": "Golay" if self.golayRadio.isChecked() else "Smooth" if self.smoothRadio.isChecked() else "Combine",
            "ind_profile": self.individualProfilesCheckbox.isChecked(),
            "total_profile": self.totalProfileCheckbox.isChecked(),
            "profile_overlay": self.profileOverlayCheckbox.isChecked(),
            "csv_table": self.csvTableCheckbox.isChecked(),
            "use_meta": self.useMeta.isChecked(),
        }

        # threading stuff
        if not self.popup_showed:
            
            self.analysis_thread = AnalysisThread(self.gui_parms)
            self.analysis_thread.progress_signal.connect(self.updateProgressBar)
            self.analysis_thread.finished.connect(self.createPlots)
            self.analysis_thread.finished.connect(self.export_gui_parms_as_html)
            self.analysis_thread.start()   
        else:
            print("Analysis aborted, incomplete parms")
            pass  


    def export_gui_parms_as_html(self):
        """
        Export the analysis as html
        """

        filename = os.path.join(self.gui_parms["output_folder"], f"{self.gui_parms['group_name']}_analysis_parameters.html")
        with document(title=f'{self.gui_parms["group_name"]} analysis') as doc:
            with doc.head:
                link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.0/css/bulma.min.css")

            h1(f'{self.gui_parms["group_name"]} analysis',_class="title")
            p(f'{datetime.now()}')
            h2('Analysis settings',_class="subtitle")
            list = ul()
            for item in self.gui_parms:
                list += li(f'{str(item)}: {str(self.gui_parms[item])}')
            h2('Plots',_class="subtitle")
            plots = os.path.join(self.gui_parms["output_folder"],f"{self.gui_parms['group_name']}_results_t_{self.gui_parms['threshold']}_f_{self.gui_parms['filter_size']}","plots")
            for image in os.listdir(plots):
                p(f"{image.rstrip('.png')}")
                div(div(figure(img(src=os.path.join(plots,image)),_class="image is-4by3"),_class="column is-half"),_class="columns is-vcentered is-centered")
            


        with open(filename, 'w') as f:
            f.write(doc.render())

        #self.show_popup(f"PDF file generated successfully: {pdf_filename}")


class AnalysisThread(QThread):
        progress_signal = pyqtSignal(int, int)  # Signal to send progress (current, total)
        finished = pyqtSignal()
        plot = pyqtSignal(str)

        def __init__(self, gui_parms):
            super().__init__()
            self.gui_parms = gui_parms
                        # Dictionnary for the output
            self.output = {'name': [], 'group': [], 'means': [],'extremum': [], 'minimum': []}     # dictionary for output
            self.total_means = []
            self.labels = []
        
        def save_to_csv(self):
            if self.gui_parms['csv_table']:
                        # save data as csv (all in one table)
                        print("Saving csv")
                        df = pd.DataFrame(data=self.output)

                        # make all the arrays start at the same location
                        for ind,array in enumerate(self.total_means):
                            self.total_means[ind] = array[np.nonzero(array)[0]]

                        # pad the arrays if not same size
                        # Find the maximum length of all arrays
                        max_length = max(len(arr) for arr in self.total_means)

                        # Pad each array to match the maximum length
                        self.total_means = [np.pad(arr, (5, max_length - len(arr)+5), mode='constant', constant_values=0) for arr in self.total_means]

                        # save all in single files
                        for name, vels in zip(self.output['name'], self.total_means):
                            try:
                                dv_axis, warn = ky.get_dv_axis(vels, self.gui_parms['dv_thresh'], self.gui_parms['pixel_size'])
                                if warn:
                                    pass
                                    # print(f"WARNING: {name} dv_axis origin is at first non-zero value")

                                df_ind = pd.DataFrame({"x-axis": dv_axis, "mean_vels": vels})
                                outdir = os.path.join(
                                    self.gui_parms['output_folder'],
                                    f"{self.gui_parms['group_name']}_results_t_{self.gui_parms['threshold']}_f_{self.gui_parms['filter_size']}"
                                )
                                ind_csv_filename = f"{name.replace('.tif', '')}.csv"

                                if not os.path.exists(outdir):
                                    os.mkdir(outdir)
                                df_ind.to_csv(os.path.join(outdir, ind_csv_filename), index=False)
                            except:
                                raise "Error"
                            
        def show_popup(self,message):
        # Create a QMessageBox
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Warning")
            msg_box.setText(message)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setStandardButtons(QMessageBox.Ok)
            self.popup_showed = True
            # Show the messagebox and handle the result
            result = msg_box.exec_()

        def run(self):
            
                for ind, path in enumerate(self.gui_parms["paths"]):
                    if os.path.isfile(os.path.normpath(path)):
                        exp = ky.Kymo(os.path.normpath(path), pixel_size=self.gui_parms["pixel_size"], frame_time=self.gui_parms["frame_time"], dv_thresh=self.gui_parms["dv_thresh"],use_metadata=self.gui_parms["use_meta"])
                        means, se = exp.generate_kymo(threshold=self.gui_parms["threshold"], thresholding_method=self.gui_parms["thresholding_method"],filtering_method=self.gui_parms["filtering_method"], save_profile=self.gui_parms["ind_profile"], filter_size=self.gui_parms["filter_size"], output_folder=self.gui_parms["output_folder"], gol_parms = self.gui_parms["gol_parms"])
                        self.total_means.append(means)
                        self.output["name"].append(exp.name.replace("_cropped","").replace(".tif",""))
                        self.output["group"].append(self.gui_parms['group_name'])
                        self.output["means"].append(means.tolist())
                        self.output["extremum"].append(np.max(means))
                        self.output["minimum"].append(np.min(means))
                        self.labels.append(exp.name)
                        del exp
                        self.progress_signal.emit(ind + 1, len(self.gui_parms["paths"]))
                        
                    else:
                        self.show_popup(f"Input Error: {path} is not valid")
                        break
                self.save_to_csv()
                self.progress_signal.emit(0, len(self.gui_parms["paths"]))
                self.finished.emit()

            


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GUI()
    sys.exit(app.exec_())

