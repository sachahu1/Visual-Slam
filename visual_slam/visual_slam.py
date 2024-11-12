import logging
import pathlib
from typing import Tuple, Sequence

import cv2
import numpy as np


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
    else:
      self._intrinsic_matrix = intrinsic_matrix

  def extract_features(self, image):
    keypoints = self._orb.detect(image, None)
    keypoints, descriptors = self._orb.compute(image, keypoints)
    return keypoints, descriptors

  def match_features(self, descriptors: Tuple[np.ndarray, np.ndarray]):
    matches = self._matcher.match(*descriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

  def camera_registration(self, points: Tuple[np.ndarray, np.ndarray]):
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
    return R, t
