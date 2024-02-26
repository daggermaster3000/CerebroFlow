
from cerebroflow.gui2 import GUI
import argparse
from pyfiglet import Figlet
import cerebroflow
import sys
from PyQt5.QtWidgets import QApplication

def main():

    parser = argparse.ArgumentParser(description=help)
    f = Figlet(font='slant')
    print(f"{f.renderText('Cerebroflow')}v.{cerebroflow.__version__}")
    print("""
                                                                            
 A tool to generate and analyze kymographs from central canal csf particle flow images.     o
                                                                                           o 
 Usage: --gui: Opens a graphical user interface                                             o
           -h: Displays this message                                                    ><'>

          
 Notes/Bugs: -Smoothing method is a combination of moving average with a Golay filter
             -Test button only works once (restart required)
             -Variablity between input images is quite high

                    
          """)
    # Add the --gui option
    parser.add_argument("--gui", action="store_true",help="Run the GUI")

    # parse the arguments
    args = parser.parse_args()

    # Check if --gui option is provided
    if args.gui:
        app = QApplication(sys.argv)
        window = GUI()
        sys.exit(app.exec_())
    else:
        print("No option provided. Use --gui to run the GUI function or -h for help")


    
if __name__ == '__main__':
    main()