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


class Kymo:
    def __init__(self,path, pixel_size, frame_time):
        self.path = path
        self.pixel_size = pixel_size    # in um
        self.frame_time = frame_time    # in s
        self.rect = {}  # rectangles for kept blobs
        self.images = []
        self.filter_size = None
        self.filtered_images = []
        self.kymo = []
        self.raw_kymo = []
        self.mean_velocities = []
        self.se_velocities = []
        self.binary_kymo = []
        # open the data
        self.data, self.name = self.open_tiff()
        # get some information about the data
        _, self.first_img = self.data.retrieve()
        self.dv_pos,self.width = np.shape(self.first_img)
        self.N_images = self.data.length
        # check the if a .npy was created if not create one
        self.init_bin()
        # convert to numpy array
        self.images = np.array(self.images,dtype='>u2')
        

    def generate_kymo(self, threshold: float, filter_size: tuple = None, thresholding_method: str = "Quantile"):
        self.threshold = threshold
        if filter_size != None:
            # if filter size is passed, filter images
            self.filtered_images = np.zeros_like(self.images)   # pre allocate to be faster
            self.filtered_images = self.filter_wiener(self.images)

            # generate kymograph
            self.kymo = self.swap_axes(self.filtered_images)
            self.raw_kymo = np.zeros_like(self.kymo)    # pre allocate to be faster
            self.raw_kymo = self.kymo.copy()    # keep a copy of the non nornalized kymograph

            # threshold 
            self.kymo = self.thresholding(self.kymo,threshold)

            # convert to binary
            self.binary_kymo = self.binary(self.kymo)

            # detect blobs
            self.velocities = self.get_velocities(self.binary_kymo)

           # process the mean and sd of all the velocities
            self.mean_velocities, self.se_velocities = self.get_mean_vel(self.velocities)

        else:
            # generate kymograph
            self.kymo = self.swap_axes(self.images)
            self.raw_kymo = np.zeros_like(self.kymo)    # pre allocate to be faster
            self.raw_kymo = self.kymo.copy()    # keep a copy of the non non-normalized kymograph

            # threshold 
            self.kymo = self.thresholding(self.kymo, threshold=threshold, method="Quantile")

            # convert to binary
            self.binary_kymo = self.binary(self.kymo)

            # detect blobs
            self.velocities = self.get_velocities(self.binary_kymo)

            # process the mean and sd of all the velocities
            self.mean_velocities, self.se_velocities = self.get_mean_vel(self.velocities)

        # show plot
        self.plot(show_plots=True, save=False)  

        return self.mean_velocities, self.se_velocities
    
    def plot(self, show_plots: bool, save: bool):
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
                plot2.title.set_text(f"2.Raw image wiener={str(self.filter_size)}")
                plot3.title.set_text("5.Kept blobs")
                plot4.title.set_text("3.Raw kymograph")
                plot5.title.set_text(f"4.Binary kymograph threshold={str(self.threshold)}")
                
                plot2.imshow(self.images[0])  
                plot4.imshow(self.raw_kymo[0])
                plot5.imshow(self.binary_kymo[0])
                plot3.imshow(self.labeled_img_array[0])
                
                # set labels
                plot1.set_xlabel(r"Dorso-ventral position [$\mu$m]")
                plot1.set_ylabel(r"Average rostro-caudal velocity [$\mu$m/s]")
                plot2.set_xlabel(r"R-C axis [frames]")
                plot2.set_ylabel(r"D-V axis [frames]")
                plot4.set_ylabel(f"Time [{self.frame_time} s]")
                plot4.set_xlabel(r"R-C axis [frames]")
                plot5.set_ylabel(f"Time [{self.frame_time} s]")
                plot5.set_xlabel(r"R-C axis [frames]")

                # plot 1
                for region in regionprops(self.labeled_img_array[0]):
                    # take regions with large enough areas
                    if (region.area < 100) and (region.area >= 15) and (region.eccentricity>0.9) and (np.degrees(region.orientation)>-95) and (np.degrees(region.orientation)<95) and (np.round(region.orientation,1)!= 0.0):         
                        # draw rectangle around good blobs
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                fill=False, edgecolor='red', linewidth=1)
                        plot3.add_patch(rect)

                # generate the x-axis in um
                dv_axis = np.arange(-(len(self.mean_velocities)-(len(self.mean_velocities)-np.nonzero(self.mean_velocities)[0][0])),len(self.mean_velocities)-np.nonzero(self.mean_velocities)[0][0])*self.pixel_size # find start of canal based on first non zero speed
                
                # plot the velocities
                plot1.plot(dv_axis,self.mean_velocities) 
                
                # Plot grey bands for the standard error
                plot1.fill_between(dv_axis, self.mean_velocities - self.se_velocities, self.mean_velocities + self.se_velocities, color='grey', alpha=0.3, label='Standard Error')
                
                # interactivity
                axtime = fig.add_axes([0.08, 0.35, 0.2, 0.03])
                time_slider = Slider(
                    ax=axtime,
                    label='Frame',
                    valmin=0,
                    valmax=self.N_images,
                    valinit=0,
                    valstep=1
                )
                # Make a vertical slider to control d-v slices
                axslice = fig.add_axes([0.03, 0.35, 0.02, 0.29])
                slice_slider = Slider(
                    ax=axslice,
                    label='d-v slice',
                    valmin=0,
                    valmax=self.dv_pos-1,
                    valinit=0,
                    valstep=1,
                    orientation="vertical"
                )
                def update(val,images=self.images):
                    plot2.imshow(images[time_slider.val])
                    
                    fig.canvas.draw_idle()

                def update_slice(val,images=self.images):
                    plot4.imshow(self.raw_kymo[slice_slider.val])
                    plot5.imshow(self.binary_kymo[slice_slider.val])
                    plot3.clear()
                    plot3.title.set_text("5.Kept blobs")
                    plot3.imshow(self.binary_kymo[slice_slider.val])
                    plot2.clear()
                    plot2.imshow(images[time_slider.val])
                    plot2.hlines(slice_slider.val,0,self.width-1, colors=['red'],label="Current slice")
                    plot2.legend()
                    for i in self.rect[slice_slider.val]:
                        plot3.add_patch(i)
                    fig.canvas.draw_idle()
                    

                # register the update function with each slider
                time_slider.on_changed(update)
                slice_slider.on_changed(update_slice)
                plt.show() 

        if save:
            print("saving figure")
            fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
            plt.style.use('Solarize_Light2')
            ax.set_xlabel(r"Dorso-ventral position [$\mu$m]")
            ax.set_ylabel(r"Average rostro-caudal velocity [$\mu$m/s]")
            dv_axis = np.arange(-(len(self.mean_velocities)-(len(self.mean_velocities)-np.nonzero(self.mean_velocities)[0][0])),len(self.mean_velocities)-np.nonzero(self.mean_velocities)[0][0])*self.pixel_size # find start of canal based on first non zero speed
            ax.plot(dv_axis,self.mean_velocities) 
            # Plot grey bands for the standard error
            ax.fill_between(dv_axis, self.mean_velocities - self.se_velocities, self.mean_velocities + self.se_velocities, color='grey', alpha=0.3, label='Standard Error')
            ax.legend()
            fig.savefig(self.name.split(".")[0]+'.png')   # save the figure to file
            plt.close(fig)    # close the figure window

    def filter_wiener(self,images: np.ndarray, filter_size: tuple):
       """
       Applies the wiener filter to the images array
       
       INPUTS:
       ------
        - images:ndarray
       PARAMETERS:
       ------
        - filter_size:tuple       
            size of the wiener filter
       """
       
       for ind,im in enumerate(images):
            print(f"Filtering image {np.round(ind/self.N_images*100,1)}%",end = "\r")
            images[ind] = wiener(im,filter_size).copy()
            self.filter_size = filter_size
       return images

    def swap_axes(self,images):
        # swap axes
        print("\nGenerating kymograph...")
        kymo = np.swapaxes(images,0,1).copy()
        return kymo

    def thresholding(self,kymo: np.ndarray, method: str, threshold):  

        # rescale the intensities
        if method == "Hardcore":
            for i in range(len(kymo)):
                kymo[i,:,:] = self.rescale(kymo[i,:,:],0,1)
                print(f"Rescaling kymograph: {np.round(i/len(kymo)*100)}%",end = "\r")
            print()
            return kymo

        if method == "Quantile":
            # find the intensity of the central canal (based on the max of the mean intensities along the dv axis)
            means = []
            dv_length = len(kymo)
            for dv in range(dv_length):
                means.append(np.mean(kymo[dv,:,:]))
            
            central_canal = [np.max(means), np.argmax(means)]   # returns the max intensity and location of th cc
            min = np.quantile(kymo[central_canal[1]], threshold) # calculate the min value based on the threshold at the cc position
            kymo[kymo < min]  = min # all values smaller than min become min
            kymo = self.rescale(kymo,1,2) # rescaling between 1 and 2

            # next we normalize the kymograph by the average value with respect to time
            avg_vs_time = np.tile(kymo.mean(axis=(0,2)),(self.width,1)).transpose()
            for i in range(kymo.shape[0]):
                kymo[i,:,:] = np.divide(kymo[i,:,:],avg_vs_time)
            
            # Next we pre-allocate some memory
            # TODO...

            # Next we perform the moving average of along the d-v axis (through the stack of kymographs)
            # This will "link" the trajectories
            self.N_avg = 3

            # TODO pre allocate to save some time
            kymo_avg = []
            for i in range(0,self.dv_pos-self.N_avg):
                kymo_avg.append(np.mean(kymo[i:i+self.N_avg,:,:],0))
            kymo_avg = np.array(kymo_avg)
            return kymo_avg.copy()
        
    def binary(self, kymo: np.ndarray):
        # Generate binary image, every pixel that is 1% above the min signal is set to 1
        binary = np.where(kymo > 1.01,1,0)
        return binary

    def get_velocities(self, binary_kymo: np.ndarray):
        # Find all the blobs
        self.labeled_img_array = []
        keepers_vel = []
        

        # iterate over every d-v pos
        for i in range(0,self.dv_pos-self.N_avg):
            good = []
            rects = []
            # detect blobs
            print(f"Detecting and processing blobs for d-v positions: {np.round(i/(self.dv_pos-self.N_avg)*100)}%",end = "\r")
            _, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(binary_kymo[i].astype(np.uint8), connectivity=8)
            # labeled_img = label(binary[i],background=0)
            self.labeled_img_array.append(labeled_img)
            
            # iterate over every blob
            for region in regionprops(labeled_img):
            # take regions with large enough areas good eccentricity and orientation
                if (region.area < 100) and (region.area >= 15) and (region.eccentricity>0.9) and (np.degrees(region.orientation)>-95) and (np.degrees(region.orientation)<95) and (np.round(region.orientation,1)!= 0.0):
                    # if good calculate the speed from the blob's orientation  Speed=1./tan(-Orient_Kymo*pi/180)*(PixelSize/FrameTime);
                    speed = (1./np.tan(-region.orientation*np.pi/180))*(self.pixel_size/self.frame_time)   # maybe un blem with the units
                    good.append(speed)
                    minr, minc, maxr, maxc = region.bbox
                    rects.append(mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                fill=False, edgecolor='red', linewidth=1))
            self.rect[i] = rects
            # if more than five events detected append to the keepers
            if len(good) < 5:
                keepers_vel.append(0)
            else:
                keepers_vel.append(good)
        print()
        return keepers_vel
    
    def get_mean_vel(self, velocities: np.ndarray):
        # compute the mean velocities and se
        mean_velocities = savgol_filter([np.mean(i) for i in velocities],10,2) # compute the mean velocities for every dv position and smooth them
        se_velocities = savgol_filter([np.std(i) / np.sqrt(np.size(i)) for i in velocities],10,2) # compute the se for every dv position and smooth them
        return mean_velocities, se_velocities

    # helper functions
    def open_tiff(self):
        """
        A function to open a .tif file
        RETURNS: a capture object and a str of the name of the image
        """
        tiff = tc.opentiff(self.path) #open img
        name = self.path.split("\\")[-1]
        
        return tiff,name

    def cache(self):
        """
        A function to save an np.array as .npy binary (not really a cache)
        """ 
        np.save("cache\\"+self.name.split(".")[0],self.images)

    def init_bin(self):
        # check the cache to see if we haven't already processed the image
        # create cache if non existent
        if "cache" not in os.listdir():
            os.mkdir("cache")
        print("Input image: ",self.name)
        # process the time lapse to array if it hasnt previously been processed
        if self.name.split(".")[0]+".npy" not in os.listdir("cache"):    
            for ind,im in enumerate(self.data):
                print(f"Processing images {np.round(ind/self.N_images*100,1)}%",end = "\r")
                self.images.append(im)
            self.cache()
        else:
            # if it already has been processed load it from the cache
            print("Loading from previous processing!")
            self.images = np.load("cache\\"+self.name.split(".")[0]+".npy",allow_pickle=True)

    def rescale(self,array,min,max):
        """
        A function that performs min-max scaling
        """
        # Perform min-max scaling
        min_val = np.min(array)
        max_val = np.max(array)
        scaled_matrix = (max-min)*(array - min_val) / (max_val - min_val) + max
        
        return scaled_matrix


def test_filter(img, name, filter_size=(5,5), threshold = 0.2, pixel_size=0.189, frame_time=0.1, show_plots=True):
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
    # swap axes
    print()
    print("Generating kymograph...")
    #kymo = np.swapaxes(images,0,2)
    kymo = np.swapaxes(images,0,1).copy()

    # rescale the intensities
    """
    for i in range(len(kymo)):
        kymo[i,:,:] = rescale(kymo[i,:,:],0,1)
        print(f"Rescaling kymograph: {np.round(i/len(kymo)*100)}%",end = "\r")
    """
    print()
   
    # find the intensity of the central canal (based on the max of the mean intensities along the dv axis)
    means = []
    dv_length = len(kymo)
    for dv in range(dv_length):
        means.append(np.mean(kymo[dv,:,:]))


    # Define initial parameters
    thresh_init = 5
    slice_init = 0

    # Create the figure and the line that we will manipulate
    plt.style.use('Solarize_Light2')
    fig, ax = plt.subplots()
    fig.suptitle("Wiener Filter Test",size=20)
    ax.text(0,0,"Move sliders to begin display")
    

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the slice.
    axslice = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(
        ax=axslice,
        label='Frames',
        valmin=0,
        valmax=N_images,
        valinit=slice_init,
        valstep=1
    )

    # Make a vertically oriented slider to control the threshold
    axthresh = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    thresh_slider = Slider(
        ax=axthresh,
        label="Filter size",
        valmin=0,
        valmax=50,
        valinit=thresh_init,
        orientation="vertical",
        valstep=1
    )


    # The function to be called anytime a slider's value changes
    def update(val,kymo=kymo,means=means,width=width):
        filter_size = (thresh_slider.val,thresh_slider.val)
        display = wiener(images[slice_slider.val],filter_size).copy()
        
        ax.clear()
        ax.imshow(display)
        fig.canvas.draw_idle()


    # register the update function with each slider
    slice_slider.on_changed(update)
    thresh_slider.on_changed(update)


    plt.show()


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
    print()
    print("Generating kymograph...")
    #kymo = np.swapaxes(images,0,2)
    kymo = np.swapaxes(images,0,1).copy()

    # rescale the intensities
    """
    for i in range(len(kymo)):
        kymo[i,:,:] = rescale(kymo[i,:,:],0,1)
        print(f"Rescaling kymograph: {np.round(i/len(kymo)*100)}%",end = "\r")
    """
    print()
   
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


def kymo1(img, name, wiener_set=False, filter_size=(5,5), threshold = 0.9, pixel_size=0.189, frame_time=0.1, show_plots=True, hardcore_thresholding=False, save=False):
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
    i = 0
    if wiener_set == True:
            for ind,im in enumerate(images):
                loop+=1
                print(f"Processing and Filtering image {np.round(ind/N_images*100,1)}%",end = "\r")
                images[ind] = wiener(im,filter_size)
                
    # swap axes
    print()
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
        # labeled_img = label(binary[i],background=0)
        labeled_img_array.append(labeled_img)
        good = []
        for region in regionprops(labeled_img):
            
        # take regions with large enough areas good eccentricity and orientation
            if (region.area < 100) and (region.area >= 15) and (region.eccentricity>0.9) and (np.degrees(region.orientation)>-95) and (np.degrees(region.orientation)<95) and (np.round(region.orientation,1)!= 0.0):
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
            plot2.title.set_text(f"2.Raw image wiener={wiener_set}")
            plot3.title.set_text("5.Kept blobs")
            plot4.title.set_text("3.Raw kymograph")
            plot5.title.set_text(f"4.Binary kymograph threshold={threshold}")
            
            plot2.imshow(first_img)  
            plot4.imshow(raw_kymo[0])
            plot5.imshow(binary[0])
            plot3.imshow(labeled_img_array[0])
            
            # set labels
            plot1.set_xlabel(r"Dorso-ventral position [$\mu$m]")
            plot1.set_ylabel(r"Average rostro-caudal velocity [$\mu$m/s]")
            plot2.set_xlabel(r"R-C axis [frames]")
            plot2.set_ylabel(r"D-V axis [frames]")
            plot4.set_ylabel(f"Time [{frame_time} s]")
            plot4.set_xlabel(r"R-C axis [frames]")
            plot5.set_ylabel(f"Time [{frame_time} s]")
            plot5.set_xlabel(r"R-C axis [frames]")
            # initial image for plot 3
            for region in regionprops(labeled_img_array[0]):
                # take regions with large enough areas
                if (region.area >= 15) and (region.eccentricity>0.9) and (np.degrees(region.orientation)>-95) and (np.degrees(region.orientation)<95) and (np.round(region.orientation,1)!= 0.0):         
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
                valmax=N_images,
                valinit=0,
                valstep=1
            )
            # Make a vertical slider to control d-v slices
            axslice = fig.add_axes([0.03, 0.35, 0.02, 0.29])
            slice_slider = Slider(
                ax=axslice,
                label='d-v slice',
                valmin=0,
                valmax=dv_pos-1,
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
                plot3.imshow(binary[slice_slider.val])
                plot2.clear()
                plot2.imshow(images[time_slider.val])
                plot2.hlines(slice_slider.val,0,width-1, colors=['red'],label="Current slice")
                plot2.legend()
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

    if save:
        print("saving figure")
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        plt.style.use('Solarize_Light2')
        ax.set_xlabel(r"Dorso-ventral position [$\mu$m]")
        ax.set_ylabel(r"Average rostro-caudal velocity [$\mu$m/s]")
        dv_axis = np.arange(-(len(mean_velocities)-(len(mean_velocities)-np.nonzero(mean_velocities)[0][0])),len(mean_velocities)-np.nonzero(mean_velocities)[0][0])*pixel_size # find start of canal based on first non zero speed
        ax.plot(dv_axis,mean_velocities) 
        # Plot grey bands for the standard error
        ax.fill_between(dv_axis, mean_velocities - se_velocities, mean_velocities + se_velocities, color='grey', alpha=0.3, label='Standard Error')
        ax.legend()
        fig.savefig(name.split(".")[0]+'.png')   # save the figure to file
        plt.close(fig)    # close the figure window

    return mean_velocities, se_velocities 
    

if __name__ == "__main__":
    path = "Z:\\qfavey\\01_Experiments\\01_CSF_FLOW\\PIPELINE_TEST\\BioProtocol_CSFflowMeasurement\\TestFiles\\FlowData\\WT5_2_cropped.tif"
    exp1 = Kymo(path, pixel_size=0.189, frame_time=0.1)
    exp1.generate_kymo(threshold=0.9)
