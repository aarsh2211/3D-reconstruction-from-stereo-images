import numpy as np
import cv2


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)


objpoints = [] # 3d point 
imgpoints = [] # 2d points 

images = cv2.imread('chess.png')
gray = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)

#for fname in gray:
   

     
ret, corners = cv2.findChessboardCorners(gray, (7,7),None)
print(ret)

   
if ret == True:
 objpoints.append(objp)

 cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
 imgpoints.append(corners)
        
         #Draw and display the corners
 cv2.drawChessboardCorners(images, (7,7), corners,ret)
 ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
 cv2.imshow('img',images)
 cv2.waitKey(0)
np.savez_compressed('b.npz', ret = ret, mtx = mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

cv2.destroyAllWindows()
