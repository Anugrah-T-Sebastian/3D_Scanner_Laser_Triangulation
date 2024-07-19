import numpy as np
import cv2 as cv
import glob
import pickle

def getCameraCalibrationValues():
    ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

    chessboardSize = (5, 5)
    frameSize = (640,480)

    folder = './ChessBoard/calibration2'


    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 5
    objp = objp * size_of_chessboard_squares_mm


    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    images = glob.glob(folder + '/*.jpg')
    pixels_per_mm = []

    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

        # Calculate the average distance between adjacent corners in pixels
        total_distance = 0
        for i in range(chessboardSize[1]):
            for j in range(chessboardSize[0] - 1):
                total_distance += np.linalg.norm(corners2[i * chessboardSize[0] + j] - corners2[i * chessboardSize[0] + j + 1])

        avg_distance_in_pixels = total_distance / (chessboardSize[1] * (chessboardSize[0] - 1))
        
        # Calculate pixels per mm
        pixels_per_mm.append( avg_distance_in_pixels / size_of_chessboard_squares_mm)
        
    else:
        print("Chessboard not found")


    cv.destroyAllWindows()
    pixels_per_mm = np.mean(pixels_per_mm)
    print(f"1 mm = {pixels_per_mm} pixels")



    ############## CALIBRATION #######################################################

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    print("Camera calibrated : \n", ret)
    print("Camera Matrix : \n", cameraMatrix)
    print("dist : \n", dist)
    print("Rotation Vectors : \n", rvecs)
    print("Translation Vectors : \n", tvecs)
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
    pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
    pickle.dump(dist, open( "dist.pkl", "wb" ))

    return pixels_per_mm, cameraMatrix, dist, rvecs, tvecs
    ############## UNDISTORTION #####################################################

    img = cv.imread('./ChessBoard/frame_00016.jpg')
    h,  w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



    # Undistort
    dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(folder +'cali1.jpg', dst)



    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(folder +'cali2.jpg', dst)




    # Reprojection Error
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )