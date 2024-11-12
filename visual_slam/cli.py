from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from visual_slam.visual_slam import VisualSlam
from visual_slam.utils.camera_calibration import calibrate


@dataclass
class Frame:
  image: np.ndarray
  keypoints: Any
  descriptors: Any


def run():
  visual_slam = VisualSlam()

  # Open the default camera
  camera = cv2.VideoCapture(0)

  previous = None
  while True:
    ret, frame = camera.read()

    keypoints, descriptors = visual_slam.extract_features(frame)

    if previous is not None:
      matches = visual_slam.match_features((descriptors, previous.descriptors))
      out = cv2.drawMatches(
        frame,
        keypoints,
        previous.image,
        previous.keypoints,
        matches,
        None,
        flags=2,
      )

      points = (
        np.float32([keypoints[m.queryIdx].pt for m in matches]),
        np.float32([previous.keypoints[m.trainIdx].pt for m in matches]),
      )
      rotation, translation = visual_slam.camera_registration(
        points,
      )
      print(rotation, translation)
    else:
      out = frame
      previous = Frame(
        image=frame,
        keypoints=keypoints,
        descriptors=descriptors,
      )

    # Display the captured frame
    cv2.imshow("Camera", out)

    keypress = cv2.waitKey(1)
    # Press 'q' to exit the loop
    if keypress == ord("q"):
      break

  # Release the capture and writer objects
  camera.release()
  cv2.destroyAllWindows()


def run_calibration():
  calibrate()
