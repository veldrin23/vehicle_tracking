from moviepy.editor import VideoFileClip
from P5_project import *

# project video
p = MajorPipe()
p.get_model()
clip1 = VideoFileClip("project_video.mp4")
clip2 = clip1.fl_image(p.pipe)
clip2.write_videofile('project_video_out.mp4', audio=False)

