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
import copy

class Kymo:
    """
    Class Constructor: Kymo
    -----------------------
    Initializes the Kymo class object.

    INPUTS:
    - path (str): Path to the image file.
    - pixel_size (float): Pixel size in micrometers.
    - frame_time (float): Frame time in seconds.
    - filter_size (tuple or None): Optional filter size for image filtering.

    OUTPUTS:
    - None
    """
    def __init__(self,path, pixel_size, frame_time, filter_size=None):
        np.seterr(all="ignore")
        self.path = path
        self.pixel_size = pixel_size    # in um
        self.frame_time = frame_time    # in s
        self.rect = {}  # rectangles for kept blobs
        self.images = []
        
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

        if filter_size != None:

            # if filter size is passed, filter images
            self.filtered_images = np.zeros_like(self.images)   # pre allocate to be faster
            self.filtered_images = self.filter_wiener(self.images, filter_size)

            # generate kymograph
            self.kymo = self.swap_axes(self.filtered_images)

        else:

            # generate kymograph
            self.kymo = self.swap_axes(self.images)
        
    def test_filter(self):
        """
        Generates an interactive plot to test the Wiener filter effects on an image slice.

        INPUTS:
        -------
        None

        OUTPUTS:
        --------
        None
        """

        # Define initial parameters
        thresh_init = 5
        slice_init = 0

        # Create the figure 
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
            valmax=self.N_images,
            valinit=slice_init,
            valstep=1
        )

        # Make a vertically oriented slider to control the filter size
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

        # initial plot
        filter_size = (thresh_init,thresh_init)
        display = wiener(self.images[slice_init],filter_size)
        ax.imshow(display)

        # The function to be called anytime a slider's value changes
        def update(val):

            if thresh_slider.val == 0:
                display = self.images[slice_slider.val].copy()
            else:
                filter_size = (thresh_slider.val,thresh_slider.val)
                display = wiener(self.images[slice_slider.val],filter_size)
            
            ax.clear()
            ax.imshow(display)
            
            fig.canvas.draw_idle()


        # register the update function with each slider
        slice_slider.on_changed(update)
        thresh_slider.on_changed(update)


        plt.show()

        
    def test_threshold(self):
        """
        Generates an interactive plot to test thresholding and binarization effects on an image slice.

        INPUTS:
        -------
        None

        OUTPUTS:
        --------
        None
        """
        # find the intensity of the central canal (based on the max of the mean intensities along the dv axis)
        means = []
        dv_length = len(self.kymo)
        for dv in range(dv_length):
            means.append(np.mean(self.kymo[dv,:,:]))

        # Define initial parameters
        thresh_init = 0.5
        slice_init = 0

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
            valmax=self.dv_pos,
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
        def update(val,kymo=self.kymo,means=means):

            central_canal = [np.max(means), np.argmax(means)]   # returns the max intensity and location of th cc
            min = np.quantile(kymo[central_canal[1]], thresh_slider.val) # calculate the min value based on the threshold at the cc position
            kymo_min = kymo.copy()[slice_slider.val]
            kymo_min[kymo_min < min]  = min # all values smaller than min become min
            kymo_min = self.rescale(kymo_min,1,2) # rescaling between 1 and 2

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

    def generate_kymo(self, threshold: float, thresholding_method = "Quantile", save_profile=False, save_display=False, filter_size = None, init_slice= 0, output_folder= None):
        """
        Performs CSF flow analysis on kymograph data.

        INPUTS:
        -------
        threshold: float
            Threshold value for thresholding.
        thresholding_method: str
            Thresholding method ("Hardcore" or "Quantile").
        save_profile: bool
            Flag to save the flow profile figure.
        save_display: bool
            Flag to save the display figure.
        filter_size: tuple or None
            Optional filter size for image filtering.
        init_slice: int
            Initial d-v slice index.

        OUTPUTS:
        --------
        mean_velocities: array
            Array of mean velocities.
        se_velocities: array
            Array of standard errors of velocities.
        """
        print(f'Analyzing {self.name}:\n')
        print(f'-threshold: \t{threshold} \n-method: \t{thresholding_method} \n-filter size: \t{filter_size}')

        self.threshold = threshold
        
        if filter_size != None:

            # if filter size is passed, filter images
            self.filtered_images = np.zeros_like(self.images)   # pre allocate
            self.filtered_images = self.filter_wiener(self.images, filter_size)

            # generate kymograph
            self.kymo = self.swap_axes(self.filtered_images).copy()
            
        else:
            self.kymo = self.swap_axes(self.images).copy()

        self.raw_kymo = np.zeros_like(self.kymo)    # pre allocate to be faster
        self.raw_kymo = self.kymo.copy()    # keep a copy of the non normalized kymograph

        # threshold 
        self.kymo = self.thresholding(self.kymo, threshold = threshold, method = thresholding_method).copy()

        # convert to binary
        self.binary_kymo = self.binary(self.kymo)

        # detect blobs
        self.velocities = self.get_velocities(self.binary_kymo)

        # process the mean and sd of all the velocities
        self.mean_velocities, self.se_velocities = self.get_mean_vel(self.velocities)

        # show plot
        self.plot(save_display=save_display, save_profile=save_profile, filter_size=filter_size, init_slice=init_slice, output_folder=output_folder) 

        print("\033[0;37;92m",end="") 
        print("Done! ")
        print("\033[0;37;40m")
        
        return self.mean_velocities, self.se_velocities
    
    def plot(self, save_profile: bool, save_display: bool, init_slice=0,filter_size=None, output_folder=None):
        """
        Sets up and displays a multi-panel figure to visualize CSF flow analysis results.

        INPUTS:
        -------
        save_profile: bool
            Flag to save the flow profile figure.
        save_display: bool
            Flag to save the display figure.
        init_slice: int
            Initial d-v slice index.
        filter_size: tuple or None
            Optional filter size for image filtering.

        OUTPUTS:
        --------
        None
        """
    
        with plt.style.context('Solarize_Light2'):
            fig = plt.figure(layout="constrained", figsize=(10,6))
            gs = GridSpec(3, 3, figure=fig)
            plot1 = fig.add_subplot(gs[0, :])
            plot2 = fig.add_subplot(gs[1, :])
            plot3 = fig.add_subplot(gs[2, -1])
            plot4 = fig.add_subplot(gs[-1, 0])
            plot5 = fig.add_subplot(gs[-1, -2])
            fig.suptitle("CSF Profiler",size=20)

            # set titles
            plot1.title.set_text("1.Flow profile")
            plot2.title.set_text(f"2.Raw image wiener={str(filter_size)}")
            plot3.title.set_text("5.Kept blobs")
            plot4.title.set_text("3.Raw kymograph")
            plot5.title.set_text(f"4.Binary kymograph threshold={str(np.round(self.threshold,1))}")
            if filter_size != None:
                plot2.imshow(self.filtered_images[init_slice])
            else:
                plot2.imshow(self.images[init_slice])  
            plot4.imshow(self.raw_kymo[init_slice])
            plot5.imshow(self.binary_kymo[init_slice])
            plot3.imshow(self.labeled_img_array[init_slice])
            
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
            for region in regionprops(self.labeled_img_array[init_slice]):
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
                
            if save_display:
                if output_folder:
                    fig.savefig(output_folder+"\\"+self.name.split(".")[0]+"_display_threshold"+str(np.round(self.threshold,1))+"_filter"+str(filter_size)+'.png',dpi=fig.dpi)
                else:
                    fig.savefig(self.name.split(".")[0]+"_display_threshold"+str(np.round(self.threshold,1))+"_filter"+str(filter_size)+'.png',dpi=fig.dpi)
                plt.close(fig)

            if save_profile:
                print("saving figure")
                fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
                plt.style.use('Solarize_Light2')
                ax.set_xlabel(r"Dorso-ventral position [$\mu$m]")
                ax.set_ylabel(r"Average rostro-caudal velocity [$\mu$m/s]")
                ax.set_title(str(self.name))
                dv_axis = np.arange(-(len(self.mean_velocities)-(len(self.mean_velocities)-np.nonzero(self.mean_velocities)[0][0])),len(self.mean_velocities)-np.nonzero(self.mean_velocities)[0][0])*self.pixel_size # find start of canal based on first non zero speed
                ax.plot(dv_axis,self.mean_velocities) 
                # Plot grey bands for the standard error
                ax.fill_between(dv_axis, self.mean_velocities - self.se_velocities, self.mean_velocities + self.se_velocities, color='grey', alpha=0.3, label='Standard Error')
                ax.legend()
                if output_folder:
                    fig.savefig(output_folder+"\\"+self.name.split(".")[0]+"_threshold"+str(np.round(self.threshold,1))+"_filter"+str(filter_size)+'.png')   # save the figure to file
                else:
                    fig.savefig(self.name.split(".")[0]+"_threshold"+str(np.round(self.threshold,1))+"_filter"+str(filter_size)+'.png')   # save the figure to file
                
                plt.close(fig)    # close the figure window

            else:
                plt.show()

    def filter_wiener(self,images: np.ndarray, filter_size: tuple):
        """
        Applies the Wiener filter to a sequence of images.

        INPUTS:
        -------
        images: ndarray
            Array of images to be filtered.
        filter_size: tuple
            Size of the Wiener filter.

        OUTPUTS:
        --------
        filtered_images: ndarray
            Filtered images after applying the Wiener filter.
        """
        out = np.zeros_like(images)
        for ind,im in enumerate(images):
            print(f"Filtering image {np.round((ind+1)/self.N_images*100,1)}%",end = "\r")
            out[ind] = wiener(im,filter_size)
        print()   
        return out

    def swap_axes(self,images):
        """
        Swaps the axes of a sequence of images to generate a kymograph.

        INPUTS:
        -------
        images: ndarray
            Array of images.

        OUTPUTS:
        --------
        kymo: ndarray
            Kymograph generated from the input images.
        """

        kymo = np.swapaxes(images,0,1).copy()
        return kymo

    def thresholding(self,kymo: np.ndarray, method: str, threshold):  
        """
        Performs thresholding and rescaling on kymograph data.

        INPUTS:
        -------
        kymo: ndarray
            Input kymograph data.
        method: str
            Thresholding method ("Hardcore" or "Quantile").
        threshold: float
            Threshold value for thresholding.

        OUTPUTS:
        --------
        kymo: ndarray
            Thresholded and normalized kymograph data.
        """

        
        if method == "Hardcore":
            # rescale the intensities directly
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
        """
        Generates a binary image from thresholded kymograph data.

        INPUTS:
        -------
        kymo: ndarray
            Thresholded kymograph data.

        OUTPUTS:
        --------
        binary: ndarray
            Binary image generated from the thresholded kymograph.
        """

        # Generate binary image, every pixel that is 1% above the min signal is set to 1
        binary = np.where(kymo > 1.01,1,0)
        return binary

    def get_velocities(self, binary_kymo: np.ndarray):
        """
        Detects and processes blobs in binary kymograph data to obtain velocities.

        INPUTS:
        -------
        binary_kymo: ndarray
            Binary kymograph data.

        OUTPUTS:
        --------
        keepers_vel: list
            List of velocities for each dorso-ventral position.
        """
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
                if (region.area >= 15) and (region.eccentricity>0.9) and (np.abs(np.sin(region.orientation))>0.1) and (np.abs(np.cos(region.orientation))>0.1):
                    # if good calculate the speed from the blob's orientation  Speed=1./tan(-Orient_Kymo*pi/180)*(PixelSize/FrameTime);
                    # note: no need to convert to rad as np takes rad directly and regionprops returns an angle in rad
                    speed = (np.arctan(-region.orientation))*(self.pixel_size/self.frame_time)   # maybe un blem with the units
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
        """
        Computes mean velocities and standard errors from a list of velocities.

        INPUTS:
        -------
        velocities: list
            List of velocities for each dorso-ventral position.

        OUTPUTS:
        --------
        mean_velocities: array
            Array of mean velocities.
        se_velocities: array
            Array of standard errors of velocities.
        """
        # compute the mean velocities and se
        mean_velocities = savgol_filter([np.mean(i) for i in velocities],20,2) # compute the mean velocities for every dv position and smooth them
        se_velocities = savgol_filter([np.std(i) / np.sqrt(np.size(i)) for i in velocities],20,2) # compute the se for every dv position and smooth them
        return mean_velocities, se_velocities

    # helper functions
    def open_tiff(self):
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
        tiff = tc.opentiff(self.path) #open img
        name = self.path.split("\\")[-1]
        
        return tiff,name

    def cache(self):
        """
        A function to save an np.array as .npy binary (not really a cache)
        """ 
        np.save("cache\\"+self.name.split(".")[0],self.images)

    def init_bin(self):
        """
        init_bin
        Initializes image data by processing time-lapse images or loading from cache if previously processed.

        INPUTS:
        -------
        None

        OUTPUTS:
        --------
        None
        """
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
        Performs min-max scaling on the input array.

        INPUTS:
        -------
        array: ndarray
            Input array for scaling.
        min: float
            Minimum value after scaling.
        max: float
            Maximum value after scaling.

        OUTPUTS:
        --------
        scaled_matrix: ndarray
            Scaled array.
        """
        # Perform min-max scaling
        min_val = np.min(array)
        max_val = np.max(array)
        scaled_matrix = (max-min)*(array - min_val) / (max_val - min_val) + max
        
        return scaled_matrix

    

if __name__ == "__main__":
    path = "Z:\\qfavey\\01_Experiments\\01_CSF_FLOW\\PIPELINE_TEST\\BioProtocol_CSFflowMeasurement\\TestFiles\\FlowData\\WT5_2_Cropped_AdditionalNoise.tif"
    exp1 = Kymo(path, pixel_size=0.189, frame_time=0.1)
    for t in np.arange(0.5,0.7,0.1):
        exp1.generate_kymo(threshold=t, save_display=True)
        for f in np.arange(1,3,1):
            exp1.generate_kymo(threshold=t, save_display=True, filter_size=(f,f))
    exp1.generate_kymo(threshold=0.6, save_display=True)
    # exp1.test_filter()
