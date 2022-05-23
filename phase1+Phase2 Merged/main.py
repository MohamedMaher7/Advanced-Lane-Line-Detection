import sys
from yolo import Process,draw_labeled_bboxes,draw
from lane import process_image
import numpy as np
from moviepy.editor import VideoFileClip


def merge(img):
    lane_img=process_image(img)
    Filtered_boxes,Filtered_confidence,Filtered_classIDs=Process(img)
    draw_img = draw_labeled_bboxes(lane_img, Filtered_boxes, Filtered_confidence, Filtered_classIDs)
    return draw_img


video_output = sys.argv[2]
input_path = sys.argv[1]

clip1 = VideoFileClip(input_path)
processed_video = clip1.fl_image(merge)
processed_video.write_videofile(video_output, audio=False)