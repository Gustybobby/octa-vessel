# config.py

# Visualization Parameters
OVERLAY_IMAGE = True  # set to overlay original vessel image over the segments
OVERLAY_INTENSITY = 0.3  # Range: 0 - 1

# Vessel Analysis Parameters
ANGLE_THRESHOLD = 135  # if < threshold then reject
MAX_DEPTH = 50  # max path length

SMALL_SEGMENT_THRESHOLD = 10  # threshold to be classify as small
NUM_POINTS = 5  # number of points to calculate angle from
MIN_FINAL_LENGTH = 50  # minimum length of the final image
SMALL_BRANCH_THRESHOLD = (
    25  # if < threshold prune out the branch (node with only 1 neighbor)
)

# Making crop TRUE will also produce a log file which contains the filename, and the coordinates of the pixels in the original image representing the vessel
CROP = True  # will save the images by cropping so only the vessel represented by the path is saved
MARGIN = 50  # is the margin after cropping the image that we will use.

MSE_THRESHOLD = 0.15  # <= to this will be considered a duplicate

NTV_THRESHOLD = 1.05
TV_THRESHOLD = 1.3
