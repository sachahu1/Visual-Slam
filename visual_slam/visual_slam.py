import logging
import pathlib
import sys
from dataclasses import dataclass
from typing import Tuple, Any, Optional

import cv2
import numpy as np
import open3d as o3d

@dataclass
class Frame:
  image: np.ndarray
  keypoints: Any
  descriptors: Any
  pose: Optional[np.ndarray] = None

class VisualSlam:
  def __init__(self, number_of_features=500, intrinsic_matrix=None):
    self._orb: cv2.ORB = cv2.ORB_create(nfeatures=number_of_features)
    self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if intrinsic_matrix is None:
      calibration_file = pathlib.Path("camera_intrinsic_matrix.txt")
      if calibration_file.exists():
        self._intrinsic_matrix = np.loadtxt(calibration_file)
      else:
        logging.error(
          "Missing camera calibration. Please provide the camera intrinsics or calibrate the camera using the `calibrate` command."
        )
        sys.exit(-1)
    else:
      self._intrinsic_matrix = intrinsic_matrix

    self.point_cloud = o3d.geometry.PointCloud()
    self._previous = None

  @property
  def previous(self):
    if self._previous is not None:
      return self._previous
    else:
      return None

  def extract_features(self, image):
    keypoints = self._orb.detect(image, None)
    keypoints, descriptors = self._orb.compute(image, keypoints)
    return keypoints, descriptors

  def match_features(self, descriptors: Tuple[np.ndarray, np.ndarray]):
    matches = self._matcher.match(*descriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

  def camera_registration(self, points: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    fundamental_matrix, inliers = cv2.findFundamentalMat(
      points[0], points[1], cv2.FM_RANSAC
    )
    essential_matrix = (
      self._intrinsic_matrix.T @ fundamental_matrix @ self._intrinsic_matrix
    )

    _, R, t, mask = cv2.recoverPose(
      essential_matrix,
      points[0][inliers],
      points[1][inliers],
      self._intrinsic_matrix,
    )

    transform = np.eye(4)
    transform[0:3, 0:3] = R
    transform[0:3, 3] = t.T
    return transform

  @staticmethod
  def add_ones(pts):
    """Helper function to add a column of ones to a 2D array (homogeneous coordinates)."""
    return np.hstack([pts, np.ones((pts.shape[0], 1))])

  def triangulate(self, pose1, pose2, pts1, pts2):
    # Initialize the result array to store the homogeneous coordinates of the 3D points
    ret = np.zeros((pts1.shape[0], 4))

    # Invert the camera poses to get the projection matrices
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)

    # Loop through each pair of corresponding points
    for i, p in enumerate(zip(self.add_ones(pts1), self.add_ones(pts2))):
      # Initialize the matrix A to hold the linear equations
      A = np.zeros((4, 4))

      # Populate the matrix A with the equations derived from the projection matrices and the points
      A[0] = p[0][0] * pose1[2] - pose1[0]
      A[1] = p[0][1] * pose1[2] - pose1[1]
      A[2] = p[1][0] * pose2[2] - pose2[0]
      A[3] = p[1][1] * pose2[2] - pose2[1]

      # Perform SVD on A
      _, _, vt = np.linalg.svd(A)

      # The solution is the last row of V transposed (V^T), corresponding to the smallest singular value
      ret[i] = vt[3]

    # Return the 3D points in homogeneous coordinates

    ret /= ret[:, 3:]
    good_pts4d = (np.abs(ret[:, 3]) > 0.005) & (ret[:, 2] > 0)

    mapp_pts = [p for i, p in enumerate(ret) if good_pts4d[i]]
    return np.array(mapp_pts)

  def add_points(self, points: np.ndarray):
    if len(self.point_cloud.points) == 0:
      self.point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    else:
      new_points = o3d.utility.Vector3dVector(points[:, :3])
      self.point_cloud.points.extend(new_points)

  def run(self):
    camera = cv2.VideoCapture(0)
    while True:
      ret, frame = camera.read()

      keypoints, descriptors = self.extract_features(frame)

      current_frame = Frame(
        image=frame,
        keypoints=keypoints,
        descriptors=descriptors,
      )

      if self.previous is None:
        current_frame.pose = np.eye(4)
        self._previous = current_frame
        continue
      else:
        matches = self.match_features((descriptors, self.previous.descriptors))
        out = cv2.drawMatches(
          frame,
          keypoints,
          self.previous.image,
          self.previous.keypoints,
          matches,
          None,
        )
        points = (
          np.float32([keypoints[m.queryIdx].pt for m in matches]),
          np.float32([self.previous.keypoints[m.trainIdx].pt for m in matches]),
        )
        transform = self.camera_registration(
          points,
        )

        current_frame.pose = np.dot(transform, self.previous.pose)

        ret = self.triangulate(
          pose1=current_frame.pose,
          pose2=self.previous.pose,
          pts1=points[0],
          pts2=points[1],
        )
        if ret.shape[0] != 0:
          self.add_points(ret)
        self._previous = current_frame

        # Display the captured frame
        cv2.imshow("Camera", out)

        keypress = cv2.waitKey(1)
        # Press 'q' to exit the loop
        if keypress == ord("q"):
          break

    # Release the capture and writer objects
    camera.release()
    cv2.destroyAllWindows()

    o3d.visualization.draw(
      [self.point_cloud],
    )
    o3d.io.write_point_cloud(
      "points_colored_structurev2.ply", self.point_cloud
    )
