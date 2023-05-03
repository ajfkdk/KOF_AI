import cv2
import numpy as np

def nothing(x):
    pass

# 创建一个空窗口
cv2.namedWindow('Threshold')

# 创建 Trackbar 来调整颜色阈值
cv2.createTrackbar('Hmin', 'Threshold', 0, 255, nothing)
cv2.createTrackbar('Smin', 'Threshold', 90, 255, nothing)
cv2.createTrackbar('Vmin', 'Threshold', 242, 255, nothing)
cv2.createTrackbar('Hmax', 'Threshold', 58, 255, nothing)
cv2.createTrackbar('Smax', 'Threshold', 255, 255, nothing)
cv2.createTrackbar('Vmax', 'Threshold', 255, 255, nothing)

while True:
    # 读取图像
    img = cv2.imread(r'C:\Users\chenj\PycharmProjects\train_your_own_game_AI\game_player\test\test3.png')

    # 将图像转换为 HSV 颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 获取 Trackbar 的值
    h_min = cv2.getTrackbarPos('Hmin', 'Threshold')
    s_min = cv2.getTrackbarPos('Smin', 'Threshold')
    v_min = cv2.getTrackbarPos('Vmin', 'Threshold')
    h_max = cv2.getTrackbarPos('Hmax', 'Threshold')
    s_max = cv2.getTrackbarPos('Smax', 'Threshold')
    v_max = cv2.getTrackbarPos('Vmax', 'Threshold')

    # 定义颜色阈值
    lower_color = np.array([0, 90, 242])
    upper_color = np.array([58, 255,255])

    # 提取颜色条的区域
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 显示结果
    cv2.imshow('Threshold', mask)

    # 等待按键事件
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 释放资源
cv2.destroyAllWindows()