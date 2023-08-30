# CerebroFlow 🧠 🐟
A tool to generate csf flow profiles based on a kymographic approach as well as other utilities 
</br>

![image](https://github.com/daggermaster3000/CerebroFlow/assets/82659911/2afe5815-18c9-40e9-95eb-1bb88d05eea1)



## Requirements
Run the following command in the console to install the requirements
```bash
pip install matplotlib PySimpleGUI opencv-python scipy scikit-image TiffCapture pandas
```

## Usage (for now)
Example code:
```python
from funcs import kymo as ky
import PySimpleGUI as sg

path = sg.popup_get_file("", no_window=True, default_extension=".tif")  # prompt the user for input file
print("Image path: ",path.replace("/","\\"))  # for windows path

exp1 = ky.Kymo(path.replace("/","\\"), pixel_size=0.189, frame_time=0.1)  # create a Kymo object

exp1.test_filter()  # open a window to test filter size
exp1.test_threshold()  # open a window to test threshold
exp1.generate_kymo(threshold=0.5)  # generate kymograph

```
### Testing parameters
#### Filter
![image](https://github.com/daggermaster3000/CerebroFlow/assets/82659911/3b8c9c81-eb35-4b50-a168-4a1e82f9ea46)
#### Threshold
![image](https://github.com/daggermaster3000/CerebroFlow/assets/82659911/0e46b671-6eb6-46e7-886c-744fbddd8ef1)


