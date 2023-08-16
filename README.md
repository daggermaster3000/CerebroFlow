# CerebroFlow 🧠 🐟
A tool to generate csf flow profiles based on a kymographic approach as well as other utilities 
</br>

![image](https://github.com/daggermaster3000/CerebroFlow/assets/82659911/3e4a240e-742e-43ef-9cda-ff7a46d60c29)

## Requirements
Run the following command in the console to install the requirements
```bash
pip install matplotlib PySimpleGUI opencv-python scipy scikit-image TiffCapture
```

## Usage (for now)
1. Run FlowJ.py
2. You will be prompted to choose a .tif file
3. CSF profiling will be ran, displaying plots and returning an array of the average velocities

### Testing parameters
Use the `ky.test_kymo_parms()` function to open a window to test different thresholds.
![image](https://github.com/daggermaster3000/CerebroFlow/assets/82659911/55f35b1f-6e1b-4d26-92e6-ce7f817b05c3)

