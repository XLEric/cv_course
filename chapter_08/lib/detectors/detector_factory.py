from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
# sys.path.append('./lib/models/')
# from .exdet import ExdetDetector
# from .ddd import DddDetector
# from .ctdet import CtdetDetector
from .multi_pose import MultiPoseDetector

detector_factory = {
  'multi_pose': MultiPoseDetector,
}
