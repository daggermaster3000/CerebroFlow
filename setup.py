from setuptools import setup

setup(
    name='cerebroflow',
    version='0.1.10',    
    description_content_type="text/plain",
    description='Cerebroflow, a package to generate csf flow profiles from csf flow experiments',
    long_description="Check the homepage for more info",
    url='https://github.com/daggermaster3000/CerebroFlow/tree/library_organisation',
    author='Quillan Favey',
    author_email='quillan.favey@gmail.com',    
    license='BSD 2-clause',
    packages=['cerebroflow'],
    install_requires=['matplotlib',
                    'PySimpleGUI',
                    'opencv-python',
                    'scipy',
                    'scikit-image',
                    'TiffCapture',
                    'pandas',
                    'numpy',
                    'tqdm',
                    'pyfiglet',
                    'dominate',
                    'plotly',
                    'aicsimageio',
            
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',       
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)  