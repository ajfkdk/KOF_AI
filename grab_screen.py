import os
import subprocess
import time
import cv2
import ffmpeg
import numpy as np
import win32gui, win32ui, win32con, win32api

# ---*---

def grab_screen(x, x_w, y, y_h):

    # 获取桌面
    hwin = win32gui.GetDesktopWindow()

    w = x_w - x
    h = y_h - y

    # 返回句柄窗口的设备环境、覆盖整个窗口，包括非客户区，标题栏，菜单，边框
    hwindc = win32gui.GetWindowDC(hwin)

    # 创建设备描述表
    srcdc = win32ui.CreateDCFromHandle(hwindc)

    # 创建一个内存设备描述表
    memdc = srcdc.CreateCompatibleDC()

    # 创建位图对象
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, w, h)
    memdc.SelectObject(bmp)
    
    # 截图至内存设备描述表
    memdc.BitBlt((0, 0), (w, h), srcdc, (x, y), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (h, w, 4)

    # 内存释放
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# ---------- 注意：以下需要设置 ----------

GAME_WIDTH   = 796    # 游戏窗口宽度
GAME_HEIGHT  = 431    # 游戏窗口高度
white_border = 51     # 游戏边框

# 是否正在录制视频
recording = False

# 视频文件保存路径
video_path = "D:/data/video/output.mp4"

# 视频编码器和帧率设置
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30.0

# 创建 VideoWriter 对象
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (GAME_WIDTH, GAME_HEIGHT))

# ---------- 注意：以上需要设置 ----------

def get_game_screen():
    return grab_screen(
        x = 165,
        x_w = GAME_WIDTH,
        y = white_border,
        y_h = white_border+GAME_HEIGHT)

# 全屏
FULL_WIDTH = 1920
FULL_HEIGHT = 1080

def get_full_screen():
    return grab_screen(
        x = 0,
        x_w = FULL_WIDTH,
        y = 0,
        y_h = FULL_HEIGHT)




import io

class VideoRecorder:
    def __init__(self, output_folder, frame_size, fps=30):
        self.output_folder = output_folder
        self.frame_size = frame_size
        self.fps = fps
        self.process = None
        self.recording = False
        self.fileName = None

    def start_recording(self):
        print("开始录制视频！")
        # 文件名为当前时间
        self.fileName = self.output_folder + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.mp4'

        # 创建 ffmpeg 视频写入器
        output_codec = 'libx264'
        output_pix_fmt = 'yuv420p'

        try:
            self.process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24',
                       s='{}x{}'.format(self.frame_size[0], self.frame_size[1]), r=self.fps, loglevel='error')
                .output(self.fileName, vcodec=output_codec, pix_fmt=output_pix_fmt, loglevel='error')
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            )
            self.recording = True
        except subprocess.SubprocessError as e:
            print(f"Failed to start ffmpeg process: {e}")
            self.recording = False
            self.process = None

    def save_record(self):
        print("取消录制视频！")
        self.recording = False
        if self.process is not None:
            self.process.communicate(input=None)
            self.process = None
            print("保存视频成功！")

    def is_recording(self):
        return self.recording

    def record(self, frame):
        if self.process and self.process.stdin:
            self.process.stdin.write(frame.tobytes())
        else:
            self.start_recording()
        # if self.recording:
        #     try:
        #
        #     except BrokenPipeError as e:
        #         print(f"Failed to write frame: {e}")
        #         self.recording = False
        #         self.cancel_recording()

    def cancel_recording(self):
        self.recording = False
        if self.process is not None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            if self.process.stdout is not None:
                self.process.stdout.close()

            try:
                self.process.communicate(timeout=5)  # 等待进程结束
            except Exception as e:
                print(f"Failed to communicate with process: {e}")
            finally:
                self.process = None
            if os.path.exists(self.fileName):
                os.remove(self.fileName)
                print("已删除录制的视频文件！")
            else:
                print("录制视频文件不存在！")
