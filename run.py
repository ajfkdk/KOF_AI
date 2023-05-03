from collections import deque
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from game_player.control_keyboard_keys import win, fail, select

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd

from game_player.brain import DoubleDQN
from game_player.others import get_status, roi
from game_player.grab_screen import get_game_screen, VideoRecorder
from game_player.detect_keyboard_keys import key_check

# ---*---

class RewardSystem:
    def __init__(self):
        self.total_reward = 0
        self.reward_history = []
        self.reward = 0

    def get_reward(self, current_status, next_status):
        if sum(next_status) == 0:
            self.reward = 0
        else:
            # 计算状态差值self 90 70
            # 受伤扣分
            self_hurt = current_status[0] - next_status[0]
            # 打人得分
            target_hurt = current_status[1] - next_status[1]
            # 防御得分：如果自己本次掉血小于10，那么就是防御到了
            if 0< self_hurt < 3:
                print(f' (◕ᴗ◕✿) 防御成功<--{self_hurt}')
                self_hurt = -10 # -10*-0.5 = 5 每次成功防御得5分


            if self_hurt == 100 or self_hurt == -100:
                self_hurt = 0
            if target_hurt == 100 or target_hurt == -100:
                target_hurt = 0
            # 加比例
            self_hurt = self_hurt * -0.6  # 自己受伤扣分

            # 对手血越少，得分越多
            ratio = 1 - (next_status[1] / 100)
            target_hurt = target_hurt * ratio * 1.2  # 打人得到的分数更多

            # 计算奖励
            self.reward = target_hurt + self_hurt
            # 计算这回合累积奖励
            self.total_reward += self.reward
            # 判断是否结束回合
            if next_status[0] == 0 or next_status[1] == 0:
                # self.total_reward不为0的话
                if self.total_reward != 0:
                    # 判断谁赢了
                    if next_status[0] == 0:
                        # print('我输了')
                        self.reward_history.append(0)  # 记录总奖励
                    else:
                        # print('我赢了')
                        self.reward_history.append(1)  # 记录总奖励
                    self.total_reward = 0  # 重置总奖励


        # print(f'  血量检查：{self.reward:>5},上一帧的自己:{current_status[0]:>5},现在的自己:{next_status[0]:>5},上一帧的对手:{current_status[1]:>5},现在的对手:{next_status[1]:>5}')
        return self.reward

    def save_reward_curve(self, save_path='reward.png'):
        total = len(self.reward_history)
        # 设置合适的图表尺寸
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为中文宋体
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号，否则会显示为方块
        plt.plot(range(total), self.reward_history)
        plt.ylabel('是否胜利')
        plt.xlabel('回合数量')
        # 生成等距的刻度
        ticks = np.linspace(0, total, num=11, dtype=int)
        plt.xticks(ticks)
        plt.axhline(y=0, color='r', linestyle='--')  # 添加红色标识线
        plt.savefig(save_path)
        print(f' 保存奖励曲线：{save_path:>5}')



# -------------------- 一些参数，根据实际情况修改 --------------------

x = 10  # 左 不小于0，小于 x_w
x_w = 626  # 右 不大于图像宽度，例如 800，大于 x
y = 94  # 上 不小于0，小于 y_h
y_h = 410  # 下不大于图像高度，例如 450，大于 y

in_depth = 1
in_height = 626  # 图像高度
in_width = 410  # 图像宽度
in_channels = 3  # 颜色通道数量
outputs = 6  # 动作数量
lr = 0.01  # 学习率

gamma = 0.80  # 越小就越注重眼前利益，越大就越注重长远利益
replay_memory_size = 500  # 记忆容量
replay_start_size = 10  # 学习了第几次之后开始自己思考怎么玩
batch_size = 50  # 样本抽取数量
update_freq = 2 # 训练评估网络的频率 第：{self.step} 次想要学习
target_network_update_freq = 2  # 更新目标网络的频率

screen_width = 796  # 图像宽度
screen_height = 431  # 图像高度

recording = True  # 是否录制视频

# -------------------- 一些参数，根据实际情况修改 --------------------

class Agent:
    # C:\Users\chenj\PycharmProjects\train_your_own_game_AI\saveMemory
    def __init__(
            self,
            save_memory_path=None,  # 指定记忆/经验保存的路径。默认为None，不保存。
            load_memory_path=None,
            save_weights_path=None,
            load_weights_path=None,
            max_steps=None,
            folder_path=None,
            vedio_folder=None,
    ):
        # 是否游戏结束
        self.game_over = False
        self.save_memory_path = save_memory_path  # 指定记忆/经验保存的路径。默认为None，不保存。
        self.load_memory_path = load_memory_path  # 指定记忆/经验加载的路径。默认为None，不加载。
        self.brain = DoubleDQN(
            in_height,  # 图像高度
            in_width,  # 图像宽度
            in_channels,  # 颜色通道数量
            outputs,  # 动作数量
            lr,  # 学习率
            gamma,  # 奖励衰减
            replay_memory_size,  # 记忆容量
            replay_start_size,  # 开始经验回放时存储的记忆量，到达最终探索率后才开始
            batch_size,  # 样本抽取数量
            update_freq,  # 训练评估网络的频率
            target_network_update_freq,  # 更新目标网络的频率
            save_weights_path,  # 指定模型权重保存的路径。默认为None，不保存。
            max_steps,  # 指定模型训练的步数。默认为None，不限制。
            folder_path,  # 指定模型保存的路径。默认为None，不保存。
            load_weights_path  # 指定模型权重加载的路径。默认为None，不加载。

        )

        # 为AI配置摄影师
        # self.recorder=VideoRecorder(vedio_folder,(screen_height,screen_width),fps=30)

        if not save_weights_path:  # 注：默认也是测试模式，若设置该参数，就会开启训练模式
            self.train = True
        else:
            self.train = True

        self.reward_system = RewardSystem()

        self.i = max_steps  # 计步器

        self.screens = deque(maxlen=in_depth * 2)  # 用双端队列存放图像

        if self.load_memory_path:
            self.load_memory()  # 加载记忆/经验

    def load_memory(self):
        if os.path.exists(self.load_memory_path):
            last_time = time.time()
            self.brain.replayer.memory = pd.read_json(self.load_memory_path)  # 从json文件加载记忆/经验。
            print(f'Load {self.load_memory_path}. Took {round(time.time() - last_time, 3):>5} seconds.')

            i = self.brain.replayer.memory.action.count()
            self.brain.replayer.i = i
            self.brain.replayer.count = i
            self.brain.step = i

        else:
            print('No memory to load.')

    def get_S(self):
        for _ in range(in_depth):
            self.screens.append(get_game_screen())  # 先进先出，右进左出

    def img_processing(self, screens):
        return np.array([cv2.resize(roi(screen, x, x_w, y, y_h), (in_height, in_width)) for screen in screens])

    def check(self):
        self.get_S()
        #     检查get_status()函数是否正确
        s1 = get_status(list(self.screens)[in_depth - 1])
        s2 = get_status(list(self.screens)[in_depth * 2 - 1])

        #     如果检查是否游戏胜利taj，失败，或者游戏误操作进入了选人界面，这里之所以使用前一个界面和后一个界面都检查一遍，是因为有时候会出现一帧的误差，导致检查不到
        if s1[2] or s1[3] or s1[4] or s2[2] or s2[3] or s2[4]:
            if s1[2] or s2[2]:
                print("赢了!")
                # self.recorder.save_record()
                self.brain.learn()  # 学习,每局学一下
                win()
            elif s1[3] or s2[3]:
                print("(o_ _)ﾉ失败)!")
                # self.recorder.cancel_recording()
                self.brain.learn()  # 学习,每局学一下
                fail()
            elif s1[4] or s2[4]:
                print("You are in the hero selection screen!")
                # self.recorder.save_record()
                self.brain.learn()  # 学习,每局学一下
                select()
            time.sleep(3)
            # self.recorder.start_recording()

    def round(self):
        # 录制视频
        if recording:
            # screens中获取最新的一帧画面
            frame = list(self.screens)[in_depth - 1]
            # self.recorder.record(frame)/

        observation = self.img_processing(list(self.screens)[:in_depth])  # 把对战画面转换为神经网络的输入格式

        action = self.action = self.brain.choose_action(observation)  # 输入对战画面，神经网络输出动作

        self.get_S()  # 观测

        reward = self.reward_system.get_reward(
            current_status=get_status(list(self.screens)[in_depth - 1]),
            next_status=get_status(list(self.screens)[in_depth * 2 - 1])
        )  # R

        next_observation = self.img_processing(list(self.screens)[in_depth:])  # S'
        self.check()
        if self.train:
            self.brain.replayer.store(
                observation,
                action,
                reward,
                next_observation
            )  # 把数据存储到经验库

    def run(self):
        paused = True
        print("Ready!")

        keys = key_check()  # 获取当前键盘按键状态

        while True:  # 进入游戏循环
            if paused:  # 如果游戏暂停
                print('\rPaused.', end='')
                if 'T' in keys:  # 如果检测到按下 'T' 键
                    self.get_S()  # 获取当前游戏状态 S
                    paused = False  # 将游戏状态改为非暂停状态
                    # self.recorder.start_recording()
                    print('\nStarting!')  # 打印游戏开始状态
            else:  # 如果游戏非暂停状态
                self.i += 1  # 记录游戏循环次数
                last_time = time.time()  # 记录上一次循环时间

                self.round()

                # 计算运行时间
                loop_time = round(time.time() - last_time, 3)

                # 打印游戏状态信息
                print(
                    f'\r {self.brain.who_play:>4} , step: {self.i:>6} . Loop took {loop_time:>5} seconds. action {self.action:>1} , total_reward: {self.reward_system.total_reward:>10.3f} , memory: {self.brain.replayer.count:7}.',
                    end='')

                if self.i % 5000 == 0:  # 每100次循环
                    self.reward_system.save_reward_curve(save_path=f"./rewardImg/reward_{self.i}.png")  # 绘制并保存奖励曲线

                if 'P' in keys:  # 如果检测到按下 'P' 键
                    self.reward_system.save_reward_curve(save_path=f"./rewardImg/reward_{self.i}.png")  # 绘制并保存奖励曲线
                    paused = True  # 将游戏状态改为暂停状态

                if 'Q' in keys:  # 如果检测到按下 'Q' 键
                    self.brain.replayer.save_memory()  # 保存记忆/经验
                    self.reward_system.save_reward_curve()  # 绘制并保存奖励曲线
                    # self.recorder.cancel_recording()//
                    print('\nQuit.')

            # 获取当前键盘按键状态
            keys = key_check()
