# CerebroFlow
A tool to generate csf flow profiles based on a kymographic approach as well as other utilities
## Dependencies
Run the following python script to install the dependencies
```python
import subprocess

def install_dependencies(dependencies):
    for dependency in dependencies:
        try:
            subprocess.check_call(['pip', 'install', dependency])
            print(f"Successfully installed {dependency}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {dependency}")

if __name__ == "__main__":
    dependencies_to_install = [
        "matplotlib",
        "PySimpleGUI",
        "opencv-python",
        "scipy",
        "scikit-image",
        "Pillow",
        "TiffCapture"
        # Probably incomplete
    ]

    install_dependencies(dependencies_to_install)

```

## Utilisation
For now run FlowJ.py...
It will run the analysis on an image of choice. There is also a test for threshold but it is incomplete
Will update page and doc later
