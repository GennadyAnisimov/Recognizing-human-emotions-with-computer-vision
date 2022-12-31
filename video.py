# -*- coding: utf-8 -*-

import cv2
import sys
from functions import emotionVideo

video_path = sys.argv[1]
if __name__ == '__main__':
    video = cv2.VideoCapture(video_path)
    emotionVideo(video)