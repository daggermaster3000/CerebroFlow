from imaris_ims_file_reader.ims import ims
import tiffcapture as tc
import numpy as np

def open_image(path):
    if path.endswith(".ims"):
        data,shape = read_ims_file(path)
    else:
         data,shape = read_tiff(path)
    name = path.split("\\")[-1]

    return data, name, shape
     
def read_ims_file(path):
    """
        Opens an ims file.

        INPUTS:
        -------
        None

        OUTPUTS:
        --------
        tiff: tc.TiffCapture
            TiffCapture object for the image file.
        name: str
            Name of the image file.
    """
    a = ims(path)
    a = a[:,0,0,:,:]
    shape = a.shape

    return a, shape


    
def read_tiff(path):
        """
        Opens a TIFF image file using the tiffcapture library.

        INPUTS:
        -------
        None

        OUTPUTS:
        --------
        tiff: tc.TiffCapture
            TiffCapture object for the image file.
        name: str
            Name of the image file.
        """
        
        tiff = tc.opentiff(path) #open img
        _, first_img = tiff.retrieve()
        shape = np.shape(first_img)
        N_images = ...
        return data, shape, N_images


