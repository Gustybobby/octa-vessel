# config.py

import logging

# Logging Configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Visualization Parameters
OVERLAY_IMAGE = True  # set to overlay original vessel image over the segments
OVERLAY_INTENSITY = 0.3  # Range: 0 - 1
SHOW_PRUNED_IMAGE = False

# Vessel Analysis Parameters
ANGLE_THRESHOLD = 30  # <= will be rejected
MAX_RECURSION_DEPTH = (
    50  # Set to None or an integer value (maximum number of nodes in a path)
)
SMALL_SEGMENT_LENGTH = 2  # pixels (<= will be considered a small segment) means they will be skipped from the calculation of the angles
NUM_POINTS = 4  # Points saved to calculate tangents (each tangent will be formed from this many continuous points)
MINIMUM_FINAL_LENGTH = 50  # segments lesser than this will not be in the final images
LEAF_BRANCH_LENGTH = 25

# Making crop TRUE will also produce a log file which contains the filename, and the coordinates of the pixels in the original image representing the vessel
CROP = True  # will save the images by cropping so only the vessel represented by the path is saved
MARGIN = 50  # is the margin after cropping the image that we will use.

REMOVE_DUPLICATES = True  # Removes duplicates at the end by comparing them through MSE
MSE_THRESHOLD = 0.15  # <= to this will be considered a duplicate
DEBUG_DUPLICATES = False  # shows each duplicate as we encounter them
