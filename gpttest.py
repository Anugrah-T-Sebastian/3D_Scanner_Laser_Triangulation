import cv2
import numpy as np

# Step 1: Load the image containing the laser scan line
image = cv2.imread('./video_frames/grey_cube_10mm/frame_00043.jpg', cv2.IMREAD_GRAYSCALE)

# Step 2: Threshold the image to create a binary image
_, binary_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

# Step 3: Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming we are interested in the largest contour
contour = max(contours, key=cv2.contourArea)

# Step 4: Fit a line to the contour points to determine the central line
rows, cols = binary_image.shape
[vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
central_line = [(0, lefty), (cols - 1, righty)]

# Step 5: Extract 3D coordinates using triangulation
# Assuming we have the camera matrix (K) and the rotation (R) and translation (T) matrices from calibration
# These are placeholders for the actual calibration data
K = np.array([[fx, 0, cx], [0, fy, cy], [0
                                         , 0, 1]])
R = np.eye(3)  # Identity matrix for simplicity
T = np.array([0, 0, -baseline])  # Translation vector for simplicity

# Example function to triangulate points
def triangulate_points(line_points, K, R, T):
    points_3d = []
    for point in line_points:
        # Convert pixel to normalized camera coordinates
        normalized_point = np.linalg.inv(K).dot([point[0], point[1], 1])
        # Perform triangulation (this is a simplified version)
        z = -T[2] / normalized_point[2]  # Depth
        x = normalized_point[0] * z
        y = normalized_point[1] * z
        points_3d.append([x, y, z])
    return np.array(points_3d)

# Create a list of points along the central line
line_points = [(x, y) for x, y in zip(np.linspace(0, cols, num=100), np.linspace(lefty, righty, num=100))]

# Triangulate the points to get the 3D point cloud
point_cloud = triangulate_points(line_points, K, R, T)

print("3D Point Cloud:", point_cloud)
