from moviepy.editor import VideoFileClip
from P5_project import *

# project video
clip1 = VideoFileClip("test_video.mp4")
clip2 = clip1.fl_image(pipe)
clip2.write_videofile('test_video_out.mp4', audio=False)


