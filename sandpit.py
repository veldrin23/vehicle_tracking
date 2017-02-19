

import sys
from moviepy.editor import VideoFileClip, TextClip


sys.path.insert(0, 'D:/carND/adv_lane_finding')

from project import *
lanes_pipe =Pipes().full_pipe
#
#
#
# clip1 = VideoFileClip("project_video.mp4")
# clip2 = clip1.fl_image()
# clip2.write_videofile('project_video_2out.mp4', audio=False)