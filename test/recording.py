import cv2
import numpy as np
import win32con
import win32gui
import win32ui

from game_player.grab_screen import VideoRecorder, grab_screen

recorder = VideoRecorder(r"D:\data\vedio\\", (640, 480), fps=30)
recorder.start_recording()

for i in range(300):
    frame = grab_screen(0, 640, 0, 480)
    recorder.record(frame)

recorder.save_record()