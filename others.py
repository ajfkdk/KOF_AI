import cv2
import numpy as np
from PIL import Image


# ---*---

def roi(img, x, x_w, y, y_h):
    return img[y:y_h, x:x_w]


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global vertices
    if event == cv2.EVENT_LBUTTONDOWN:
        vertices.append([x, y])
        try:
            cv2.imshow("window", img)
        except NameError:
            pass
    return vertices


def get_xywh(img):
    global vertices
    vertices = []

    print('Press "ESC" to quit. ')
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("window", on_EVENT_LBUTTONDOWN)
    while True:
        cv2.imshow("window", img)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    if len(vertices) != 4:
        print("vertices number not match")
        return -1

    x = min(vertices[0][0], vertices[1][0])
    x_w = max(vertices[2][0], vertices[3][0])
    y = min(vertices[1][1], vertices[2][1])
    y_h = max(vertices[0][1], vertices[3][1])

    cv2.imshow('img', img)
    cv2.imshow('roi(img)', roi(img, x, x_w, y, y_h))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'\n x={x}, x_w={x_w}, y={y}, y_h={y_h}\n')


# ---------- 以下需要修改 ----------

def get_Self_HP(img):
    img = roi(img, x=56, x_w=240, y=21, y_h=31)
    canny = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 0, 100)
    value = canny.argmax(axis=-1)
    return np.median(value)


def get_Target_HP(img):
    img_roi = roi(img, x=391, x_w=575, y=20, y_h=30)
    img_roi = cv2.medianBlur(img_roi, 5)
    canny = cv2.Canny(cv2.GaussianBlur(img_roi, (5, 5), 0), 50, 100)
    value = canny.argmax(axis=-1)
    return value


def get_Self_HP2(img):
    # 将图像转换为 HSV 颜色空间
    img = roi(img, x=56, x_w=240, y=21, y_h=31)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义颜色阈值
    lower_color = np.array([0, 90, 242])
    upper_color = np.array([58, 255, 255])

    # 提取颜色条的区域
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 进行形态学运算，填充颜色条的空洞和去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel)

    # 显示血条的区域
    # cv2.imshow('self', mask)

    # 计算颜色条的长度
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        color_bar_length = w
    else:
        color_bar_length = 0

    # 根据颜色条长度与血条总长度的比例来估算血量
    total_bar_length = 183  # 假设血条总长度为 240 像素
    if color_bar_length > 0:
        hp_percent = color_bar_length / total_bar_length
        hp_value = int(hp_percent * 100)  # 假设最大血量为 100
    else:
        hp_value = 0

    return hp_value


def get_Target_HP2(img):
    # 将图像转换为 HSV 颜色空间
    img = roi(img, x=391, x_w=575, y=20, y_h=30)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义颜色阈值
    lower_color = np.array([0, 90, 242])
    upper_color = np.array([58, 255, 255])

    # 提取颜色条的区域
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 进行形态学运算，填充颜色条的空洞和去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel)

    # 显示血条的区域
    # cv2.imshow('target', mask)

    # 计算颜色条的长度
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        color_bar_length = w
    else:
        color_bar_length = 0

    # 根据颜色条长度与血条总长度的比例来估算血量
    total_bar_length = 183  # 假设血条总长度为 240 像素
    if color_bar_length > 0:
        hp_percent = color_bar_length / total_bar_length
        hp_value = int(hp_percent * 100)  # 假设最大血量为 100
    else:
        hp_value = 0

    return hp_value


def isGameWin(img):
    img_roi = roi(img, x=153, x_w=279, y=15, y_h=62)
    # cv2.imshow('img_roi', img_roi)
    template = cv2.imread(
        r'C:\Users\chenj\PycharmProjects\train_your_own_game_AI\game_player\template\win.png')
    img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    threshold = 0.9
    res = cv2.matchTemplate(img_roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > threshold:
        return 1
    else:
        return 0


def isGameLose(img):
    img_roi = roi(img, x=320, x_w=499, y=155, y_h=232)
    template = cv2.imread(
        r'C:\Users\chenj\PycharmProjects\train_your_own_game_AI\game_player\template\lose.png')
    img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    threshold = 0.9
    res = cv2.matchTemplate(img_roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > threshold:
        return 1
    else:
        return 0


# 是否是选人画面
def isSelectHero(img):
    img_roi = roi(img, x=237, x_w=410, y=1, y_h=57)
    # template = cv2.imread('./template/heroSelector.jpg')
    template = cv2.imread(
        r'C:\Users\chenj\PycharmProjects\train_your_own_game_AI\game_player\template\heroSelector.jpg')
    img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    threshold = 0.9
    res = cv2.matchTemplate(img_roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val > threshold:
        return 1
    else:
        return 0


# 是否是

# 不够就自己添加，多了就自己删除

def get_status(img):
    return np.array((get_Self_HP2(img), get_Target_HP2(img), isGameWin(img), isGameLose(img), isSelectHero(img)))
# ---------- 以上需要修改 ----------
