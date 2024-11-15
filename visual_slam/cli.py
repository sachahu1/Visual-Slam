from visual_slam.utils.camera_calibration import calibrate
from visual_slam.visual_slam import VisualSlam


def run():
  visual_slam = VisualSlam()
  visual_slam.run()

def run_calibration():
  calibrate()
