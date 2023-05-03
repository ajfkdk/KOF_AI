

import cv2
import numpy as np

from game_player.grab_screen import get_game_screen, grab_screen

from game_player.others import get_xywh, isSelectHero, roi, isGameWin, isGameLose

# x=237, x_w=410, y=1, y_h=57
# 加载模板
template = cv2.imread('../template/heroSelector.jpg')

# 设置模板匹配的阈值
threshold = 0.5
def test1():
    while True:
        screen = get_game_screen()
        result = isGameLose(screen)

        cv2.waitKey(1)
def selectWindow():
    print("开始选择区域")
    screen = get_game_screen()
    get_xywh(screen)

test1()