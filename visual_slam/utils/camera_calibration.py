import pathlib
import shutil

import cv2
import numpy as np


def capture_image(out_path: pathlib.Path):
  # Open the default camera
  cam = cv2.VideoCapture(0)

  i = 0
  while True:
    ret, frame = cam.read()

    # Display the captured frame
    cv2.imshow("Camera", frame)

    # Write the frame to the output file
    cv2.imwrite((out_path / f"{i}.png").as_posix(), frame)

    i += 1
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord("q"):
      break

  # Release the capture and writer objects
  cam.release()
  cv2.destroyAllWindows()


def calibrate():
  """Checkerboard camera calibration.

  Checkerboard camera calibration inspired by https://learnopencv.com/camera-calibration-using-opencv/.
  This function takes a number of images from the camera, identifies a
  checkerboard and attempts to calibrate the camera. If successful, the function
  writes the intrinsic camera parameters to a file.
  """
  calibration_folder = pathlib.Path("calibration")
  calibration_folder.mkdir(exist_ok=True)

  capture_image(calibration_folder)

  # Defining the dimensions of checkerboard
  CHECKERBOARD = (7, 7)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # Creating vector to store vectors of 3D points for each checkerboard image
  objpoints = []
  # Creating vector to store vectors of 2D points for each checkerboard image
  imgpoints = []

  # Defining the world coordinates for 3D points
  objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
  objp[0, :, :2] = np.mgrid[
    0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]
  ].T.reshape(-1, 2)

  for fname in calibration_folder.iterdir():
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
      gray,
      CHECKERBOARD,
      cv2.CALIB_CB_ADAPTIVE_THRESH
      + cv2.CALIB_CB_FAST_CHECK
      + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not ret:
      continue

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
      objpoints.append(objp)
      # refining pixel coordinates for given 2d points.
      corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

      imgpoints.append(corners2)

      # Draw and display the corners
      img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

  cv2.destroyAllWindows()

  shutil.rmtree(calibration_folder)

  h, w = img.shape[:2]

  """
  Performing camera calibration by 
  passing the value of known 3D points (objpoints)
  and corresponding pixel coordinates of the 
  detected corners (imgpoints)
  """
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
  )

  print("Camera matrix : \n")
  print(mtx)
  print("dist : \n")
  print(dist)
  print("rvecs : \n")
  print(rvecs)
  print("tvecs : \n")
  print(tvecs)

  np.savetxt("camera_intrinsic_matrix.txt", mtx)
