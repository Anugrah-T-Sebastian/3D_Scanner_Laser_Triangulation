import open3d as o3d
import numpy as np

def getPointClouds(depthMap, cameraIntrinsic):
    height, width = depthMap.shape
    points = []

    for v in range(height):
        for u in range(width):
            Z = depthMap[v, u]
            if Z == 0:  # Skip invalid depth values
                continue
            X = (u - cameraIntrinsic['cx']) * Z / cameraIntrinsic['fx']
            Y = (v - cameraIntrinsic['cy']) * Z / cameraIntrinsic['fy']
            points.append([X, Y, Z])

    points = np.array(points)

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

# Example usage
depthMap = np.array([
    [1000, 1010, 1020, 1030, 1040],
    [1050, 1060, 1070, 1080, 1090],
    [1100, 1110, 1120, 1130, 1140],
    [1150, 1160, 1170, 1180, 1190],
    [1200, 1210, 1220, 1230, 1240]
], dtype=np.float32)

cameraIntrinsic = {
    'fx': 1000.0,
    'fy': 1000.0,
    'cx': 2.0,
    'cy': 2.0
}

pcd = getPointClouds(depthMap, cameraIntrinsic)

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])
