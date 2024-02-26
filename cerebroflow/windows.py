import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QTabWidget, QListWidget, QListWidgetItem, QMenu, QGraphicsView, QLabel, QColorDialog
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtCore import Qt
import mplcursors  # Import mplcursors

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        # Enable drag-and-drop for the main window
        self.setAcceptDrops(True)

        # Create a button to open the file dialog
        self.button = QPushButton('Open Folder', self)
        self.button.clicked.connect(self.open_folder_dialog)

        # Create a tab widget to manage tabs
        self.tab_widget = QTabWidget()

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.tab_widget)

        # Set the layout for the main window
        self.setLayout(layout)

        # Set window properties
        self.setWindowTitle('Main Window')
        self.setGeometry(100, 100, 600, 400)

    def open_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')

        if folder_path:
            # Create a tab for the selected folder
            tab = FolderTab(folder_path)
            self.tab_widget.addTab(tab, os.path.basename(folder_path))
            self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1 and event.mimeData().urls()[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        folder_path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(folder_path):
            # Create a tab for the dropped folder
            tab = FolderTab(folder_path)
            self.tab_widget.addTab(tab, os.path.basename(folder_path))
            self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)

class FolderTab(QWidget):
    def __init__(self, folder_path):
        super().__init__()

        self.folder_path = folder_path
        self.csv_data = None
        self.plot_window = None

        self.init_ui()

    def init_ui(self):
        # Create a list widget to display CSV files
        self.list_widget = QListWidget(self)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.list_widget)

        # Set the layout for the tab
        self.setLayout(layout)

        # Set window properties
        self.setWindowTitle(os.path.basename(self.folder_path))
        self.setGeometry(200, 200, 400, 300)

        # Update the list content
        self.update_list_content()

        # Connect custom context menu to the list widget
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.show_context_menu)
                

    def update_list_content(self):
        # Get all CSV files in the folder and its subfolders
        csv_files = [os.path.join(root, file) for root, dirs, files in os.walk(self.folder_path) for file in files if file.endswith('.csv')]

        # Clear the list
        self.list_widget.clear()

        # Populate the list with full paths of CSV files
        for csv_path in csv_files:
            item = QListWidgetItem(csv_path)
            self.list_widget.addItem(item)

    def show_context_menu(self, pos):
        # Create a context menu
        menu = QMenu(self)

        # Add actions to the context menu
        action1 = menu.addAction("Plot CSV")

        # Show the context menu at the given position
        action = menu.exec_(self.list_widget.mapToGlobal(pos))

        # Handle selected action
        if action == action1:
            selected_item = self.list_widget.itemAt(pos)
            selected_csv_path = selected_item.text()

            # Load CSV data
            self.load_csv_data(selected_csv_path)

            # Open a new window for plotting
            self.open_plot_window()

    def load_csv_data(self, csv_path):
        try:
            self.csv_data = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error loading CSV data: {e}")

    def open_plot_window(self):
        if self.csv_data is not None:
            self.plot_window = PlotWindow(self.csv_data, self.folder_path, self.list_widget)
            self.plot_window.show()

class PlotWindow(QWidget):
    def __init__(self, csv_data, folder_path,list_widget):
        super().__init__()

        self.csv_data = csv_data
        self.folder_path = folder_path
        self.list_widget = list_widget

        self.init_ui()

    def init_ui(self):
        # Create a color dialog button
        self.color_button = QPushButton("Select Plot Color", self)
        self.color_button.clicked.connect(self.show_color_dialog)

        # Create a color dialog button
        self.add_button = QPushButton("Add selection to plot", self)
        self.add_button.clicked.connect(self.add_data_to_plot)

        # Create a matplotlib figure and canvas for plotting
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.figure)

        # Create a QLabel for displaying hover information
        self.hover_label = QLabel(self)
        self.hover_label.setAlignment(Qt.AlignCenter)
        self.hover_label.setText("Hover to show postion, r-click on label to remove")

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.color_button)
        layout.addWidget(self.add_button)
        layout.addWidget(self.canvas)
        layout.addWidget(self.hover_label)

        # Set the layout for the window
        self.setLayout(layout)

        # Set window properties
        self.setWindowTitle("CSV Plot Window")
        self.setGeometry(400, 400, 800, 600)

        # Plot initial data
        self.plot_data()

        # Enable cursor hovering
        self.cursor = mplcursors.cursor(hover=True)
        

    def plot_data(self):
        if self.csv_data is not None:
            self.ax.clear()
            line = self.ax.plot(self.csv_data['x-axis'], self.csv_data['mean_vels'])[0]
            self.ax.set_xlabel("dorso-ventral axis [um]")
            self.ax.set_ylabel("Velocity [um/s]")
            
            self.canvas.draw()

    def add_data_to_plot(self):
        selected_items = self.list_widget.selectedItems()
        
        self.load_csv_data(selected_items[0].text())
        
        self.ax.plot(self.csv_data['x-axis'], self.csv_data['mean_vels'])
        self.canvas.draw()
        
    def show_color_dialog(self):
        color_dialog = QColorDialog(self)
        color = color_dialog.getColor()

        if color.isValid():
            # Set the line color
            self.ax.lines[0].set_color(color.name())
            self.canvas.draw()
    
    def load_csv_data(self, csv_path):
        try:
            self.csv_data = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error loading CSV data: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create an instance of the MainWindow class
    main_window = MainWindow()

    # Show the main window
    main_window.show()

    sys.exit(app.exec_())
