
from cerebroflow.gui import GUI
import argparse
from pyfiglet import Figlet
import cerebroflow

def main():

    parser = argparse.ArgumentParser(description=help)
    f = Figlet(font='slant')
    print(f"{f.renderText('Cerebroflow')}v.{cerebroflow.__version__}")
    print("""

 A tool to generate and analyze kymographs from central canal csf particle flow images.

 Usage: --gui: Opens a graphical user interface
           -h: Displays this message

          
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
        
        Gui = GUI()
        Gui.start()
    else:
        print("No option provided. Use --gui to run the GUI function or -h for help")


    
if __name__ == '__main__':
    main()