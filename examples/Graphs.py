from funcs import kymo as ky
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import matplotlib.lines as mlines
from skimage.measure import label, regionprops

# Prompt user to choose file


import os


# Prompt user to choose file


path = sg.popup_get_file("", no_window=True, default_extension=".tif")
path = os.path.normpath(path)
print("Image path: ",path)
# pixel size: 0.16250000000000003 or 0.189 (pub)
pixel_size = 0.16250000000000003
frame_time = 0.159
exp1 = ky.Kymo(path, pixel_size=pixel_size, frame_time=frame_time)
exp1.generate_kymo(threshold=0.8,dash=False)

kymos = exp1.labeled_img_array.copy()

speed_image = np.zeros_like(kymos,dtype=float)

for id, kymo in enumerate(kymos):
    
    for region in regionprops(kymo):
        # take regions with large enough areas
        if (region.area < 100) and (region.area >= 15) and (region.eccentricity>0.9) and (np.degrees(region.orientation)>-95) and (np.degrees(region.orientation)<95) and (np.round(region.orientation,1)!= 0.0):         
            speed = (np.tan(-region.orientation))*(pixel_size/frame_time) 
            for l,c in zip(range(0,np.shape(speed_image[id])[0]),range(0,np.shape(speed_image[id])[1])):
                for coords in region.coords:
                    #print(l,c,coords[0],coords[1])
                    if l == coords[0] and c == coords[1]:
                        #print(speed)
                        speed_image[id][l][c] = speed
    print(np.max(speed_image[id]))             
                   
tagged = np.swapaxes(speed_image,1,2)
print(np.shape(tagged))
print(np.shape(speed_image))
plt.imshow(tagged[:,:,155],cmap="magma")
plt.show()
