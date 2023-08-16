"""
A function to get the kymograph of an image file 
and analyse it to generate a csf flow profile (based on Thouvenin et al. 2020)
"""

import cv2
import tiffcapture as tc
import numpy as np
from scipy.signal import wiener, savgol_filter
import matplotlib.pyplot as plt
import os
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, TextBox
import time

# helper functions
def open_tiff(Path):
    """
    A function to open a .tif file
    RETURNS: a capture object and a str of the name of the image
    """
    tiff = tc.opentiff(Path) #open img
    name = Path.split("\\")[-1]
    
    return tiff,name

def cache(image,name):
    """
    A function to save an np.array as .npy binary
    """
   
    np.save("cache\\"+name.split(".")[0],image)

def rescale(array,min,max):
    """
    A function that performs min-max scaling
    """
    # Perform min-max scaling
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_matrix = (max-min)*(array - min_val) / (max_val - min_val) + max
    
    return scaled_matrix

def test_kymo_parms(img, name, wiener, filter_size=(5,5), threshold = 0.2, pixel_size=0.189, frame_time=0.1, show_plots=True):
    """
    A function that lets the user test out different parameters
    INPUTS:
        - img
        - name
    OUTPUTS:
        - array of means and se velocities
        - plots 
    PARAMETERS:
            name        type        function
        - wiener        bool        defines whether wiener filter is applied
        - filter_size   tuple       kernel size of the wiener filter
        - pixel_size    float       pixel size of input img [um]
        - frame_time    float       time betweem frames [s]
    """
    np.seterr(all='ignore') # ignore numpy warnings
    
    # some variables
    loop=0
    images = []

    # get first image of the sequence and height and width
    _, first_img = img.retrieve()
    dv_pos,width = np.shape(first_img)
    N_images = img.length
    
    # check the cache to see if we haven't already processed the image
    # create cache if non existent
    if "cache" not in os.listdir():
        os.mkdir("cache")
    print("Input image: ",name)
    # process the time lapse to array if it hasnt previously been processed
    if name.split(".")[0]+".npy" not in os.listdir("cache"):    
        for im in img:
            loop+=1
            print(f"Processing image {np.round(loop/N_images*100,1)}%",end = "\r")
            images.append(im)
        cache(images,name)
    else:
        # if it already has been processed load it from the cache
        print("Loading from previous processing!")
        images = np.load("cache\\"+name.split(".")[0]+".npy")
    
    # convert to numpy array
    images = np.array(images,dtype='>u2')
    if wiener == True:
            for im,ind in enumerate(images):
                loop+=1
                print(f"Processing and Filtering image {np.round(i/N_images*100,1)}%",end = "\r")
                images[id] = wiener(im,filter_size)
    # swap axes
    print("Generating kymograph...")
    #kymo = np.swapaxes(images,0,2)
    kymo = np.swapaxes(images,0,1).copy()

    # rescale the intensities
    """
    for i in range(len(kymo)):
        kymo[i,:,:] = rescale(kymo[i,:,:],0,1)
        print(f"Rescaling kymograph: {np.round(i/len(kymo)*100)}%",end = "\r")
    print()
    """
    # find the intensity of the central canal (based on the max of the mean intensities along the dv axis)
    means = []
    dv_length = len(kymo)
    for dv in range(dv_length):
        means.append(np.mean(kymo[dv,:,:]))


    # Define initial parameters
    thresh_init = 0.5
    slice_init = 159

    # Create the figure and the line that we will manipulate
    plt.style.use('Solarize_Light2')
    fig, ax = plt.subplots()
    fig.suptitle("Test threshold",size=20)
    ax.text(0,0,"Move sliders to begin display")
    

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the slice.
    axslice = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(
        ax=axslice,
        label='d-v slice',
        valmin=0,
        valmax=300,
        valinit=slice_init,
        valstep=1
    )

    # Make a vertically oriented slider to control the threshold
    axthresh = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    thresh_slider = Slider(
        ax=axthresh,
        label="Threshold",
        valmin=0,
        valmax=1,
        valinit=thresh_init,
        orientation="vertical"
    )


    # The function to be called anytime a slider's value changes
    def update(val,kymo=kymo,means=means,width=width):

        central_canal = [np.max(means), np.argmax(means)]   # returns the max intensity and location of th cc
        min = np.quantile(kymo[central_canal[1]], thresh_slider.val) # calculate the min value based on the threshold at the cc position
        kymo_min = kymo.copy()[slice_slider.val]
        kymo_min[kymo_min < min]  = min # all values smaller than min become min
        kymo_min = rescale(kymo_min,1,2) # rescaling between 1 and 2

        # next we normalize the kymograph by the average value with respect to time
        avg_vs_time = kymo_min.mean().transpose()
        kymo_min = np.divide(kymo_min,avg_vs_time)

        # Generate binary image every pixel that is 1% above the min signal is equal to 1
        binary = np.where(kymo_min > 1.01,1,0)
        ax.clear()
        ax.imshow(binary)
        fig.canvas.draw_idle()


    # register the update function with each slider
    slice_slider.on_changed(update)
    thresh_slider.on_changed(update)


    plt.show()


def kymo1(img, name, wiener, filter_size=(5,5), threshold = 0.9, pixel_size=0.189, frame_time=0.1, show_plots=True, hardcore_thresholding=False):
    """
    A function that generates a csf flow profile based on a kymographic approach

    Inputs:
    ----------
        - img:capture object
        - name:str

    Outputs:
    ----------
        - array of means and se velocities
        - plots 

    Parameters:
    ----------
    wiener:bool 
            defines whether wiener filter is applied
    filter_size:tuple 
            kernel size of the wiener filter
    pixel_size:float 
            pixel size of input img [um]
    frame_time:float
            time betweem frames [s]
    hardcore thresholding:bool
            rescales between 0 and 1 then applies the thresholding (pretty good when loads of noise)
    """
    np.seterr(all='ignore') # ignore numpy warnings
    
    # some variables
    loop=0      # for displaying progress bar
    images = []

    # get first image of the sequence and height and width
    _, first_img = img.retrieve()
    dv_pos,width = np.shape(first_img)
    N_images = img.length

    # check the cache to see if we haven't already processed the image
    # create cache if non existent
    if "cache" not in os.listdir():
        os.mkdir("cache")
    print("Input image: ",name)

    # process the time lapse to array if it hasnt previously been processed
    if name.split(".")[0]+".npy" not in os.listdir("cache"):    
        for im in img:
                loop+=1
                print(f"Processing image {np.round(loop/N_images*100,1)}%",end = "\r")
                images.append(im)
        cache(images,name)
    else:
        # if it already has been processed load it from the cache
        print("Loading from previous processing!")
        images = np.load("cache\\"+name.split(".")[0]+".npy")
    
    # convert to numpy array
    images = np.array(images,dtype='>u2')
    if wiener == True:
            for im,ind in enumerate(images):
                loop+=1
                print(f"Processing and Filtering image {np.round(i/N_images*100,1)}%",end = "\r")
                images[id] = wiener(im,filter_size)
    # swap axes
    print("Generating kymograph...")
    #kymo = np.swapaxes(images,0,2).copy()
    kymo = np.swapaxes(images,0,1).copy()
    raw_kymo = kymo.copy()
    #print("Kymograph shape: ",np.shape(kymo))

    # rescale the intensities
    if hardcore_thresholding:

        for i in range(len(kymo)):
            kymo[i,:,:] = rescale(kymo[i,:,:],0,1)
            print(f"Rescaling kymograph: {np.round(i/len(kymo)*100)}%",end = "\r")
        print()
    
    # find the intensity of the central canal (based on the max of the mean intensities along the dv axis)
    means = []
    dv_length = len(kymo)
    for dv in range(dv_length):
        means.append(np.mean(kymo[dv,:,:]))
    
    central_canal = [np.max(means), np.argmax(means)]   # returns the max intensity and location of th cc
    min = np.quantile(kymo[central_canal[1]], threshold) # calculate the min value based on the threshold at the cc position
    kymo[kymo < min]  = min # all values smaller than min become min
    kymo = rescale(kymo,1,2) # rescaling between 1 and 2

    # next we normalize the kymograph by the average value with respect to time
    avg_vs_time = np.tile(kymo.mean(axis=(0,2)),(width,1)).transpose()
    for i in range(kymo.shape[0]):
        kymo[i,:,:] = np.divide(kymo[i,:,:],avg_vs_time)
    
    # Next we pre-allocate some memory
    # TODO...

    # Next we perform the moving average of along the d-v axis (through the stack of kymographs)
    # This will "link" the trajectories
    N_avg = 3

    # TODO pre allocate to save some time
    kymo_avg = []
    for i in range(0,dv_pos-N_avg):
        kymo_avg.append(np.mean(kymo[i:i+N_avg,:,:],0))

    # Generate binary image every pixel that is 1% above the min signal is equal to 1
    kymo_avg = np.array(kymo_avg)
    binary = np.where(kymo_avg > 1.01,1,0)
     

    # Find all the blobs
    labeled_img_array = []
    keepers_vel = []
    for i in range(0,dv_pos-N_avg):
        print(f"Detecting and processing blobs for d-v positions: {np.round(i/(dv_pos-N_avg)*100)}%",end = "\r")
        _, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(binary[i].astype(np.uint8), connectivity=8)
        #labeled_img = label(binary[i],background=0)
        labeled_img_array.append(labeled_img)
        good = []
        for region in regionprops(labeled_img):
            
        # take regions with large enough areas good eccentricity and orientation
            if (region.area < 200) and (region.area >= 15) and (region.eccentricity>0.9) and (np.degrees(region.orientation)>-95) and (np.degrees(region.orientation)<95) and (np.round(region.orientation,1)!= 0.0):
                # if good calculate the speed from the blob's orientation  Speed=1./tan(-Orient_Kymo*pi/180)*(PixelSize/FrameTime);
                speed = (1./np.tan(-region.orientation*np.pi/180))*(pixel_size/frame_time)/100   # maybe un blem with the units
                good.append(speed)
   
        # if more than five events detected append to the keepers
        if len(good) < 5:
            keepers_vel.append(0)
        else:
            keepers_vel.append(good)
    print()
    # compute the mean velocities and se
    mean_velocities = savgol_filter([np.mean(i) for i in keepers_vel],10,2) # compute the mean velocities for every dv position and smooth them
    se_velocities = savgol_filter([np.std(i) / np.sqrt(np.size(i)) for i in keepers_vel],10,2)
    
    # plotting
    # setup figure
    if show_plots:
        with plt.style.context('Solarize_Light2'):
            fig = plt.figure(layout="constrained")
            gs = GridSpec(3, 3, figure=fig)
            plot1 = fig.add_subplot(gs[0, :])
            plot2 = fig.add_subplot(gs[1, :])
            plot3 = fig.add_subplot(gs[2, -1])
            plot4 = fig.add_subplot(gs[-1, 0])
            plot5 = fig.add_subplot(gs[-1, -2])
            fig.suptitle("CSF Profiler",size=20)

            # set titles
            plot1.title.set_text("1.Flow profile")
            plot2.title.set_text(f"2.Raw image wiener={wiener}")
            plot3.title.set_text("5.Kept blobs")
            plot4.title.set_text("3.Raw kymograph")
            plot5.title.set_text(f"4.Binary kymograph threshold={threshold}")
            
            plot2.imshow(first_img)  
            plot4.imshow(raw_kymo[159])
            plot5.imshow(binary[159])
            plot3.imshow(labeled_img_array[159])
            
            # set labels
            plot1.set_xlabel(r"Dorso-ventral position [$\mu$m]")
            plot1.set_ylabel(r"Average rostro-caudal velocity [$\mu$m/s]")
            plot2.set_xlabel(r"R-C axis [frames]")
            plot2.set_ylabel(r"D-V axis [frames]")
            plot4.set_ylabel(f"Time [{frame_time} s]")
            plot4.set_xlabel(r"R-C axis [frames]")
            plot5.set_ylabel(f"Time [{frame_time} s]")
            plot5.set_xlabel(r"R-C axis [frames]")
            for region in regionprops(labeled_img_array[159]):
                # take regions with large enough areas
                if (region.area < 200) and (region.area >= 15) and (region.eccentricity>0.9) and (np.degrees(region.orientation)>-95) and (np.degrees(region.orientation)<95) and (np.round(region.orientation,1)!= 0.0):         
                    # draw rectangle around good blobs
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=1)
                    plot3.add_patch(rect)
                dv_axis = np.arange(-(len(mean_velocities)-(len(mean_velocities)-np.nonzero(mean_velocities)[0][0])),len(mean_velocities)-np.nonzero(mean_velocities)[0][0])*pixel_size # find start of canal based on first non zero speed
                plot1.plot(dv_axis,mean_velocities) 
                # Plot grey bands for the standard error
                plot1.fill_between(dv_axis, mean_velocities - se_velocities, mean_velocities + se_velocities, color='grey', alpha=0.3, label='Standard Error')

            # Make a horizontal slider to control the time.
            axtime = fig.add_axes([0.08, 0.35, 0.2, 0.03])
            time_slider = Slider(
                ax=axtime,
                label='Frame',
                valmin=0,
                valmax=300,
                valinit=0,
                valstep=1
            )
            # Make a vertical slider to control d-v slices
            axslice = fig.add_axes([0.03, 0.35, 0.02, 0.29])
            slice_slider = Slider(
                ax=axslice,
                label='d-v slice',
                valmin=0,
                valmax=300,
                valinit=0,
                valstep=1,
                orientation="vertical"
            )
            def update(val,images=images):
                plot2.imshow(images[time_slider.val])
                fig.canvas.draw_idle()

            def update_slice(val,images=images):
                plot4.imshow(raw_kymo[slice_slider.val])
                plot5.imshow(binary[slice_slider.val])
                plot3.clear()
                plot3.title.set_text("5.Kept blobs")
                plot3.imshow(labeled_img_array[slice_slider.val])
                for region in regionprops(labeled_img_array[slice_slider.val]):
                    # take regions with large enough areas
                    if (region.area >= 15) and (region.eccentricity>0.9) and (np.degrees(region.orientation)>-95) and (np.degrees(region.orientation)<95) and (np.round(region.orientation,1)!= 0.0):         
                        # draw rectangle around good blobs
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                fill=False, edgecolor='red', linewidth=1)
                        plot3.add_patch(rect)
                fig.canvas.draw_idle()

            # register the update function with each slider
            time_slider.on_changed(update)
            slice_slider.on_changed(update_slice)
            plt.show() 

    return mean_velocities, se_velocities 
    

if __name__ == "__main__":
    path = "Z:\\qfavey\\01_Experiments\\01_CSF_FLOW\\PIPELINE_TEST\\BioProtocol_CSFflowMeasurement\\TestFiles\\FlowData\\WT5_2_cropped.tif"
    data,name = open_tiff(path)
    kymo1(data,name,wiener=False)
    #test_kymo_parms(data,name,wiener=False)
