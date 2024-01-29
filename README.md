# CerebroFlow
A tool to generate csf flow profiles based on an automatic kymograph analysis approach.
</br>

![image](https://github.com/daggermaster3000/CerebroFlow/assets/82659911/2afe5815-18c9-40e9-95eb-1bb88d05eea1)



## Installation
To install just run
```bash
pip install cerebroflow
```

## Usage 

### Using the GUI
To use cerebroflow with the gui, run the following in your python environment.
```bash
python -m cerebroflow --gui
```
Run analysis will output individual flow profiles as well as the mean flow profile of all the analyzed images and a csv file containing the data.
</br>
</br>
<img width="617" alt="Screenshot 2024-01-29 at 09 24 07" src="https://github.com/daggermaster3000/CerebroFlow/assets/82659911/1f7fda5f-ef79-499a-b729-4aa7a0e1ad81">
<img width="794" alt="Screenshot 2024-01-29 at 09 23 53" src="https://github.com/daggermaster3000/CerebroFlow/assets/82659911/3f5ca22d-1bf0-4099-acb7-fd03d838b53a">



### If you want to code
Check the [examples](https://github.com/daggermaster3000/CerebroFlow/tree/library_organisation/examples) folder for some graphs and other stuff.


Example code:
```python
from funcs import kymo as ky
import PySimpleGUI as sg

path = sg.popup_get_file("", no_window=True, default_extension=".tif")  # prompt the user for input file

exp1 = ky.Kymo(path, pixel_size=0.189, frame_time=0.159)  # create a Kymo object

exp1.test_filter()  # open a window to test filter size
exp1.test_threshold()  # open a window to test threshold
exp1.generate_kymo(threshold=0.5)  # generate kymograph

```
### Testing parameters
You can also test some of the parameters.
#### Threshold
<img width="752" alt="Screenshot 2024-01-29 at 09 31 12" src="https://github.com/daggermaster3000/CerebroFlow/assets/82659911/6f0d88e6-c347-44fc-b111-c8702678e10d">

#### Filter
This implementation uses a wiener filter to remove noise but it is not very successful. N2V denoising works much better, I will maybe implement it if I have time.



