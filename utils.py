import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, interp1d
from video_test import extract_frames
import open3d as o3d

# ANSI escape codes for some colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # Resets the color to default

def cropFrame(frame):
    print("Original shape:",frame.shape)
    frame = frame[:, :]
    return frame

def applyGaussianBlur(frame):
    # frame = cv2.GaussianBlur(frame, (7, 7), 0)
    # frame = cv2.GaussianBlur(frame, (7, 7), 0)
    # frame = cv2.GaussianBlur(frame, (7, 7), 0)
    return frame


def gradientTests(frame):
    lap = cv2.Laplacian(frame, cv2.CV_64F, ksize=3)
    lap = np.uint8(np.absolute(lap))
    sobelX = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(frame, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)

    maxIntensity = np.argmax(sobelX, axis=1)
    edgeTrace = np.zeros(frame.shape)
    for i in range(frame.shape[0]):
        edgeTrace[i][maxIntensity[i]] = 255
    # print("Edge Trace Test:", edgeTrace)

    titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined', 'edgeTrace']

    images = [frame, lap, sobelX, sobelY, sobelCombined, edgeTrace]
    for i in range(6):
        plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    # plt.show()
    return maxIntensity, sobelX

def getSobelXGradient(frame):
    sobelX = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
    return sobelX

def getIntensityIndex(y_values):
    y_values = abs(y_values)
    # print("Y values")
    # print(abs(y_values))
    x_values = np.arange(len(y_values))

    # Define the Gaussian function
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

    # Initial guess for the parameters: amplitude, mean, stddev
    initial_guess = [max(y_values), np.argmax(y_values), 1]

    # Perform the curve fitting
    try:
        params, covariance = curve_fit(gaussian, x_values, y_values, p0=initial_guess, maxfev=10000)
    except RuntimeError as e:
        print(f"{Colors.FAIL}Curve fitting failed: {e}{Colors.ENDC}")
        return np.argmax(y_values)

    # Extract the fitted parameters
    fitted_amplitude, fitted_mean, fitted_stddev = params

    # Find the index closest to the mean
    closest_index = np.argmin(np.round(np.abs(x_values - fitted_mean)))
    # print("Closest index to the mean:", closest_index)

    # Plot the original data and the fitted Gaussian curve
    # plt.plot(x_values, y_values, 'b-', label='data')
    # plt.plot(x_values, gaussian(x_values, *params), 'r--', label='fit: mean=%5.3f' % fitted_mean)
    # plt.axvline(x=fitted_mean, color='g', linestyle='--', label='Mean (µ)')
    # plt.axvline(x=closest_index, color='m', linestyle='--', label='Closest Index')
    # plt.xlabel('Index')
    # plt.ylabel('Y-values')
    # plt.legend()
    # plt.show()
    return closest_index

import numpy as np

def split_groups(values, threshold):
    sorted_values = sorted(values)
    
    # Find the split point for the highest group
    split_index = len(sorted_values)
    for i in range(len(sorted_values) - 1, 0, -1):
        if sorted_values[i] - sorted_values[i - 1] > threshold:
            split_index = i
            break
    
    # Identify the high values group
    high_values = sorted_values[split_index:]
    # print(f"{Colors.OKGREEN}High Values:{Colors.ENDC}")
    # print(high_values)
    
    # Create the groups maintaining the original order and length
    group1 = []
    group2 = []
    high_values_set = set(high_values)
    used_high_values = set()
    
    for value in values:
        if value in high_values_set:
            group1.append(value)
            group2.append(None)  # Placeholder for missing values
            used_high_values.add(value)  # Mark the value as used to handle duplicates
        else:
            group1.append(None)  # Placeholder for missing values
            group2.append(value)
    
    # print(f"{Colors.OKGREEN}Group 1:{Colors.ENDC}")
    # print(group1)
    # print(f"{Colors.OKGREEN}Group 2:{Colors.ENDC}")
    # print(group2)
    
    # Get the indices and values for the known points in group1
    known_indices = [i for i, x in enumerate(group1) if x is not None]
    known_values = [x for x in group1 if x is not None]
    
    # Perform linear interpolation to fill the missing values
    interp_values = np.round(np.interp(range(len(group1)), known_indices, known_values)).astype(int)
    group1 = [interp_values[i] if x is None else x for i, x in enumerate(group1)]
    
    return group1, group2

def split_groups_polynomial(values, threshold, degree=3):
    sorted_values = sorted(values)
    
    # Find the split point for the highest group
    split_index = len(sorted_values)
    for i in range(len(sorted_values) - 1, 0, -1):
        if sorted_values[i] - sorted_values[i - 1] > threshold:
            split_index = i
            break
    
    high_values = sorted_values[split_index:]
    
    group1 = []
    group2 = []
    high_values_set = set(high_values)
    used_high_values = set()
    
    for value in values:
        if value in high_values_set:
            group1.append(value)
            group2.append(None)
            used_high_values.add(value)
        else:
            group1.append(None)
            group2.append(value)
    
    known_indices = [i for i, x in enumerate(group1) if x is not None]
    known_values = [x for x in group1 if x is not None]
    
    poly_interp = np.poly1d(np.polyfit(known_indices, known_values, degree))
    interp_values = np.round(poly_interp(range(len(group1)))).astype(int)
    
    group1 = [interp_values[i] if x is None else x for i, x in enumerate(group1)]
    
    return group1, group2

def split_groups_spline(values, threshold):
    sorted_values = sorted(values)
    
    # Find the split point for the highest group
    split_index = len(sorted_values)
    for i in range(len(sorted_values) - 1, 0, -1):
        if sorted_values[i] - sorted_values[i - 1] > threshold:
            split_index = i
            break
    
    high_values = sorted_values[split_index:]
    
    group1 = []
    group2 = []
    high_values_set = set(high_values)
    used_high_values = set()
    
    for value in values:
        if value in high_values_set:
            group1.append(value)
            group2.append(None)
            used_high_values.add(value)
        else:
            group1.append(None)
            group2.append(value)
    
    known_indices = [i for i, x in enumerate(group1) if x is not None]
    known_values = [x for x in group1 if x is not None]
    
    cs = CubicSpline(known_indices, known_values)
    interp_values = np.round(cs(range(len(group1)))).astype(int)
    
    group1 = [interp_values[i] if x is None else x for i, x in enumerate(group1)]
    
    return group1, group2

def split_groups_nearest(values, threshold):
    sorted_values = sorted(values)
    
    # Find the split point for the highest group
    split_index = len(sorted_values)
    for i in range(len(sorted_values) - 1, 0, -1):
        if sorted_values[i] - sorted_values[i - 1] > threshold:
            split_index = i
            break
    
    high_values = sorted_values[split_index:]
    
    group1 = []
    group2 = []
    high_values_set = set(high_values)
    used_high_values = set()
    
    for value in values:
        if value in high_values_set:
            group1.append(value)
            group2.append(None)
            used_high_values.add(value)
        else:
            group1.append(None)
            group2.append(value)
    
    known_indices = [i for i, x in enumerate(group1) if x is not None]
    known_values = [x for x in group1 if x is not None]
    
    interp_values = np.round(np.interp(range(len(group1)), known_indices, known_values, left=known_values[0], right=known_values[-1])).astype(int)
    
    group1 = [interp_values[i] if x is None else x for i, x in enumerate(group1)]
    
    return group1, group2


# Example usage

def getReferenceFrame():
    referenceFrames = extract_frames("grey_cube_20mm", ".mp4")
    intensity_list = []

    calibrationValues = []
    
    for frame in referenceFrames:
        croppedFrame = cropFrame(frame)
        blurredFrame = applyGaussianBlur(croppedFrame)
        edgeTrace = getSobelXGradient(blurredFrame)
        intensity = np.array([getIntensityIndex(i) for i in edgeTrace])
        
        # for i in edgeTrace:
        #     intensity.append(getIntensityIndex(abs(i)))
        # intensity = np.array(intensity)
        # print(f"{Colors.OKGREEN}Intensity:{Colors.ENDC}")
        # print(intensity)
        # Append intensity to the list
        referenceLine, pixelLine  = split_groups(intensity, 5)
        print(f"{Colors.OKGREEN}Refernce Line (linear interpolation):{Colors.ENDC}")
        print(np.var(referenceLine))
        referenceLine, pixelLine  = split_groups_polynomial(intensity, 5, 2)

        print(f"{Colors.OKGREEN}Refernce Line (Polynomial Interpolation):{Colors.ENDC}")
        print(np.var(referenceLine))
        referenceLine, pixelLine  = split_groups_spline(intensity, 5)
        print(f"{Colors.OKGREEN}Refernce Line (Cubic Spline Interpolation):{Colors.ENDC}")
        print(np.var(referenceLine))
        referenceLine, pixelLine  = split_groups_nearest(intensity, 5)
        print(f"{Colors.OKGREEN}Refernce Line (Nearest Neighbor Interpolation):{Colors.ENDC}")
        print(np.var(referenceLine))
        print(f"\n{Colors.OKGREEN}Pixel Line:{Colors.ENDC}")
        print(pixelLine)
        referenceLine = np.array(referenceLine)
        pixelDisplacement = abs(intensity - referenceLine)
        pixelDisplacement = [i for i in pixelDisplacement if i != 0]
        # print(f"{Colors.OKGREEN}Pixel Displacement:{Colors.ENDC}")
        # print(pixelDisplacement)
        print(f"{Colors.OKGREEN}Mean Pixel Displacement:{Colors.ENDC}")
        print(np.mean(pixelDisplacement))
        calibrationValues.append(np.mean(pixelDisplacement))
        
        # You can add more processing or conditions here if needed
        
        # Uncomment if you want to break after processing the first frame
        break
        
    # Convert intensity_list to a NumPy array
    print(f"{Colors.OKGREEN}Calibration Value:{Colors.ENDC}")
    print(np.mean(calibrationValues))
    
    # print("Reference Frame shape:", referenceFrame.shape)
    # meanReference = np.mean(referenceFrame, axis=0).astype(int)
    # print("Mean reference shape:", meanReference.shape)
    # print("Mean Reference:", meanReference)
    # print("Reference Frame:", referenceFrame)

def getReferenceLine(frame):
    croppedFrame = cropFrame(frame)
    blurredFrame = applyGaussianBlur(croppedFrame)
    edgeTrace = getSobelXGradient(blurredFrame)
    intensity = []
    for i in edgeTrace:
        intensity.append

    intensity = np.array([getIntensityIndex(i) for i in edgeTrace])
    # print(f"{Colors.OKGREEN}Intensity:{Colors.ENDC}")
    # print(intensity)


    # Append intensity to the list
    # referenceLine, pixelLine  = split_groups(intensity, 5)
    # print(f"{Colors.OKGREEN}Refernce Line (linear interpolation):{Colors.ENDC}")
    # print(referenceLine)
    referenceLine, pixelLine  = split_groups_polynomial(intensity, 5)
    # print(f"{Colors.OKGREEN}Refernce Line (Polynomial Interpolation):{Colors.ENDC}")
    # print(referenceLine)
    # referenceLine, pixelLine  = split_groups_spline(intensity, 5)
    # print(f"{Colors.OKGREEN}Refernce Line (Cubic Spline Interpolation):{Colors.ENDC}")
    # print(referenceLine)
    # referenceLine, pixelLine  = split_groups_nearest(intensity, 5)
    # print(f"{Colors.OKGREEN}Refernce Line (Nearest Neighbor Interpolation):{Colors.ENDC}")
    # print(referenceLine)
    # print(f"\n{Colors.OKGREEN}Pixel Line:{Colors.ENDC}")
    # print(pixelLine)
   
    print(f"\n{Colors.OKGREEN}Pixel displacement:{Colors.ENDC}")
    print(abs(intensity - referenceLine))

    depth_map = np.zeros(frame.shape)
    for i in range(frame.shape[0]):
        depth_map[i][referenceLine[i]] = abs(intensity[i] - referenceLine[i])
    
    return depth_map


frame = cv2.imread("./video_frames/grey_step_8.94mm/frame_00003.jpg", cv2.IMREAD_GRAYSCALE)
depth_data = getReferenceLine(frame)

# print(f"\n{Colors.OKGREEN}Pixel Line:{Colors.ENDC}")
# print(depth_data)


# Visualize the depth map
plt.imshow(depth_data)
plt.title('Depth Map')
plt.colorbar()
plt.show()

# getReferenceFrame()



def getPointClouds(depthMap, cameraIntrinsic):
    height, width = depthMap.shape
    points = []

    fx, fy = cameraIntrinsic[0, 0], cameraIntrinsic[1, 1]
    cx, cy = cameraIntrinsic[0, 2], cameraIntrinsic[1, 2]

    for v in range(height):
        for u in range(width):
            Z = depthMap[v, u]
            if Z == 0:  # Skip invalid depth values
                continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append([X, Y, Z])

    points = np.array(points)
    print(f"{Colors.FAIL}Points:{Colors.ENDC}")
    print(points)
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def getIntrinsics(H, W, fov = 70.0):
    # Calculate the focal length
    f = W / (2 * np.tan(np.deg2rad(fov) / 2))
    # Calculate the intrinsic matrix
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    return K
cameraIntrinsic = getIntrinsics(480, 640)
pcd = getPointClouds(depth_data, cameraIntrinsic)

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])