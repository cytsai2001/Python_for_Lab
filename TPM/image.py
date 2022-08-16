# -*- coding: utf-8 -*-
"""
1. Localization:
(a) get average image of N_loc pictures
(b) get contours using Canny edge detection algorithm
(c) get edges of contours, and use image moment of edges to get center of positions, (x,y)
(d) get avg. intensity of each aoi, and remove aoi which avg. intensity < blacklevel
(e) sort (x,y) of each aoi according to distance between y-axis(x=0)
(f) select one aoi of each cluster. cluster: all aoi which distance < criteria_dist
(g) fit each aoi with 2D Gaussian to get accurate (x,y)
(h) draw aoi circle and show(save or not) figure to 'output.png'

"""
### import used modules first
from TPM.BinaryImage import BinaryImage
from basic.select import select_folder
from basic.decorator import timing

### parameters for localization
frame_read_forcenter = 0  # no need to change, frame to autocenter beads
N_loc = 10  # number of frame to stack and localization
contrast = 3

put_text = False
criteria_dist = 800  # beabs are closer than 'criteria_dist' will remove
aoi_size = 5
blacklevel = 40
whitelevel = 200
low = 40
high = 120

### parameters for tracking
read_mode = 1 # mode = 0 is only calculate 'frame_setread_num' frame, other numbers(default) present calculate whole glimpsefile
frame_setread_num = 50 # only useful when mode = 0, can't exceed frame number of a file
frame_start = 0 ## starting frame for tracking

@timing
def localization(path_folder, read_mode, frame_setread_num, frame_start, criteria_dist,
                 aoi_size, frame_read_forcenter,N_loc, contrast, low, high,
                 blacklevel, whitelevel, put_text):

    ### Localization
    Glimpse_data = BinaryImage(path_folder, read_mode=read_mode, frame_setread_num=frame_setread_num,
                               frame_start=frame_start,criteria_dist=criteria_dist, aoi_size=aoi_size,
                                frame_read_forcenter=frame_read_forcenter,N_loc=N_loc, contrast=contrast,
                                low=low, high=high,blacklevel=blacklevel,whitelevel=whitelevel,
                               )
    bead_radius, random_string = Glimpse_data.Localize(put_text=put_text) # localize beads

    return Glimpse_data, bead_radius, random_string

if __name__ == "__main__":
    path_folder = select_folder()
    Glimpse_data, bead_radius, random_string = localization(path_folder, read_mode, frame_setread_num, frame_start, criteria_dist,
                                             aoi_size, frame_read_forcenter,N_loc, contrast, low, high,
                                             blacklevel, whitelevel, put_text)

    cX = Glimpse_data.cX
    cY = Glimpse_data.cY

