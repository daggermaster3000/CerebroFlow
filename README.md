# CerebroFlow
A tool to generate csf flow profiles based on an automatic kymograph analysis approach.
</br>

<img width="757" alt="terminal image" src="https://github.com/daggermaster3000/CerebroFlow/assets/82659911/78bd6876-96be-4d15-b8d8-577b9c4d4cc5">




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
</br>
</br>
This should open the following window: 
<img width="812" alt="image" src="https://github.com/daggermaster3000/CerebroFlow/assets/82659911/6b554f46-47e4-4cca-a80e-a5c2925c261a"></br>
Add the path to your data as well as an output path. Next adjust your settings for the analysis in the `settings` tab.
<img width="812" alt="image" src="https://github.com/daggermaster3000/CerebroFlow/assets/82659911/e24f4d7c-2605-4c72-9335-f28dbb12d58c"></br>
Finally don't forget to name your output file.
<img width="812" alt="image" src="https://github.com/daggermaster3000/CerebroFlow/assets/82659911/1343d2f3-2af5-43f5-a7e3-190ad8031bb4"></br>
`Run analysis` will output individual flow profiles as well as the mean flow profile of all the analyzed images and a csv file containing the data.


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



