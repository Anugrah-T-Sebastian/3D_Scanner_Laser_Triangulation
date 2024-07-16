import numpy as np
import cv2
import matplotlib.pyplot as plt

# Example depth data in millimeters (replace with your actual depth data)
depth_data = np.array([
    [1000, 1010, 1020, 1030, 1040],
    [1050, 1060, 1070, 1080, 1090],
    [1100, 1110, 1120, 1130, 1140],
    [1150, 1160, 1170, 1180, 1190],
    [1200, 1210, 1220, 1230, 1240]
], dtype=np.float32)

# Normalize the depth map for visualization (optional)
depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
depth_normalized = depth_normalized.astype(np.uint8)

# Visualize the depth map
plt.imshow(depth_normalized, cmap='gray')
plt.title('Depth Map')
plt.colorbar()
plt.show()
