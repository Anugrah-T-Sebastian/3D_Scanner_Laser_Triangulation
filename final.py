import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from video_test import extract_frames
from utils import cropFrame, applyGaussianBlur, gradientTests, getReferenceFrame

# ANSI escape codes for some colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # Resets the color to default




meanReference, referenceFrame = getReferenceFrame()

#  VIDEO COMPUTATION
pixelDisplacement, meanPixelDisplacement = getPixelDisplacement("grey_cube_10mm", meanReference, 355, 0, 500, ".mp4")
# pixelDisplacement, meanPixelDisplacement = getPixelDisplacement("grey_cube_20mm", meanReference, 360, 110, 130)
# pixelDisplacement, meanPixelDisplacement = getPixelDisplacement("grey_cube_30mm", meanReference, 360, 220, 250)
# pixelDisplacement, meanPixelDisplacement = getPixelDisplacement("grey_step_1.48mm", meanReference, 370, 0, 500, ".mp4")
# pixelDisplacement, meanPixelDisplacement = getPixelDisplacement("grey_step_3.68mm", meanReference, 370, 0, 500, ".mp4")
# pixelDisplacement, meanPixelDisplacement = getPixelDisplacement("grey_step_8.94mm", meanReference, 370, 0, 500, ".mp4")
# pixelDisplacement, meanPixelDisplacement = getPixelDisplacement("grey_step_14mm", meanReference, 360, 0, 500, ".mp4")
print("Mean reference:", meanReference)
print("Pixel Displacement:", meanPixelDisplacement)

# FRAME COMPUTATION
# frame = cv2.imread("./video_frames/grey_cube_20mm/frame_00050.jpg", cv2.IMREAD_GRAYSCALE)
# frame = cv2.imread("./video_frames/grey_cube_30mm/frame_00050.jpg", cv2.IMREAD_GRAYSCALE)
# frame = cv2.imread("./video_frames/grey_step_8.94mm/frame_00012.jpg", cv2.IMREAD_GRAYSCALE)
# croppedFrame = cropFrame(frame)
# blurredFrame = applyGaussianBlur(croppedFrame)
# intensity, edgeTrace = gradientTests(blurredFrame)
# filteredIntensity = [value for value in intensity if abs(value) <= 360]    # Filter out values that are less than 10
# pixelDisplacement  =  abs(filteredIntensity - meanReference)
# print("Intesity:", intensity)
# print("Filtered Intensity:", filteredIntensity)
# pixelDisplacement = [value for value in pixelDisplacement if abs(value) >= 370]    # Filter out values that are less than 10
# pixelDisplacement = abs(np.mean(pixelDisplacement))
# print("Mean Reference:", meanReference)
# print("Pixel Disparity:", pixelDisplacement)
# cv2.imshow("edge Frame", edgeTrace)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cubeFrames = extract_frames("grey_cube")


