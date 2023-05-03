# https://github.com/ZhiqingXiao/rl-book/blob/master/chapter06_approx/MountainCar-v0_tf.ipynb
# https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/DQN3

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import os
import pandas as pd
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(tf.config.experimental.get_device_details(gpus[0])['device_name'])
    #     限制显存使用量
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],  # 对0号gpu进行限制
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)]
    )

# ---------- 以下根据 control_keyboard_keys.py 里定义的函数来导入 ----------
from game_player.control_keyboard_keys import J, D, L, A, D_J, A_J


# ---------- 以上根据 control_keyboard_keys.py 里定义的函数来导入 ----------

# ---*---

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(
            index=range(capacity),
            columns=['observation', 'action', 'reward', 'next_observation']
        )
        self.i = 0  # 行索引
        self.count = 0  # 经验存储数量
        self.capacity = capacity  # 经验容量

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity  # 更新行索引
        self.count = min(self.count + 1, self.capacity)  # 保证数量不会超过经验容量

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)


# ---*---

# DoubleDQN
class DoubleDQN:
    def __init__(
            self,
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
            save_weights_path,  # 指定模型权数保存的路径。
            step,  # 训练步数
            folder_path,  # 指定模型权数保存的路径。
            load_weights_path  # 指定模型权重加载的路径。

    ):
        self.in_height = in_height
        self.in_width = in_width
        self.in_channels = in_channels
        self.outputs = outputs
        self.lr = lr,
        self.gamma = gamma
        self.folder_path = folder_path
        self.replay_memory_size = replay_memory_size
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size

        self.update_freq = update_freq
        self.target_network_update_freq = target_network_update_freq

        self.save_weights_path = save_weights_path
        self.load_weights_path = load_weights_path

        self.min_epsilon = 0.2  # 最终探索率有20%的概率随机选择动作

        self.evaluate_net = self.build_network()  # 评估网络
        self.target_net = self.build_network()  # 目标网络
        self.replayer = DQNReplayer(self.replay_memory_size)  # 经验回放

        self.step = step  # 计步

    # 评估网络和目标网络的构建方法
    def build_network(self):
        input_shape = [self.in_height, self.in_width, self.in_channels]

        inputs = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
        x = tf.cast(inputs, tf.float32)
        outputs = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, weights=None, classes=self.outputs)(x)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.CategoricalCrossentropy(),  # 你觉得有更好的可以自己改
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )

        if self.load_weights_path:
            if os.path.exists(self.load_weights_path):
                model.load_weights(self.load_weights_path)
                print('Load ' + self.load_weights_path)
            else:
                print('Nothing to load')

        return model

    # 行为选择方法
    def choose_action(self, observation):

        # 先看运行的步数(self.step)有没有达到开始回放经验的要求(self.replay_start_size)，没有就随机探索
        # 如果已经达到了，就再看随机数在不在最终探索率范围内，在的话也是随机探索
        if self.step <= self.replay_start_size or np.random.rand() < self.min_epsilon:
            q_values = np.random.rand(self.outputs)
            self.who_play = '随机探索'
        else:
            observation = observation.reshape(-1, self.in_height, self.in_width, self.in_channels)
            q_values = self.evaluate_net.predict(observation)[0]
            self.who_play = '模型预测'

        action = np.argmax(q_values)

        # ---------- 以下根据 control_keyboard_keys.py 里定义的函数来修改 ----------

        """
        将所有的动作都编码成数字，并且数字满足从零开始和正整数的要求。
        例如
            W 移动 前 0
            S 移动 后 1
            A 移动 左 2
            D 移动 右 3
        """

        # 执行动作
        if action == 0:
            J()
        elif action == 1:
            A()
        elif action == 2:
            D()
        elif action == 3:
            L()
        elif action == 4:
            D_J(0.1)
        elif action == 5:
            A_J(0.1)
        # 不够可以添加，注意，一定要是正整数，还要和上一个相邻
        # ---------- 以上根据 control_keyboard_keys.py 里定义的函数来修改 ----------

        return action

    # 学习方法
    def learn(self, verbose=0):
        self.step += 1
        print(f'第：{self.step} 次想要学习')
        # 当前步数满足更新评估网络的要求
        if self.step % self.update_freq == 0:
            # 当前步数满足更新目标网络的要求
            if self.step % self.target_network_update_freq == 0:
                print(f'目标网络的学习(＾Ｕ＾)：{self.step}')
                self.update_target_network()

            print(f'----------------------------ᕙ༼ ͝°益° ༽ᕗ 开始学习：{self.step}')
            # 经验回放
            observations, actions, rewards, next_observations = self.replayer.sample(self.batch_size)

            # 数据预处理
            observations = observations.reshape(-1, self.in_height, self.in_width, self.in_channels)
            actions = actions.astype(np.int8)
            next_observations = next_observations.reshape(-1, self.in_height, self.in_width, self.in_channels)

            next_eval_qs = self.evaluate_net.predict(next_observations)
            next_actions = next_eval_qs.argmax(axis=-1)

            next_qs = self.target_net.predict(next_observations)
            next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions]

            us = rewards + self.gamma * next_max_qs
            targets = self.evaluate_net.predict(observations)
            targets[np.arange(us.shape[0]), actions] = us

            self.evaluate_net.fit(observations, targets, batch_size=8, epochs=10, verbose=verbose)

            self.save_evaluate_network()
            print(f'----------------------------------学习结束(●—●)：{self.step}')

    # 更新目标网络权重方法
    def update_target_network(self):
        self.target_net.set_weights(self.evaluate_net.get_weights())

    # 保存评估网络权重方法
    def save_evaluate_network(self):
        # save_weights_path为空的话就保存为D:\data\训练文件\train_0.h5
        if not self.save_weights_path:
            self.save_weights_path = os.path.join(self.folder_path, 'train_0.h5')
        if (self.step % 10 == 0):
            # save_weights_path=D:\data\训练文件\train_0.h5 ，找到train_的位置，然后后面加上step + .h5
            self.save_weights_path = self.save_weights_path.split('_')[0] + '_' + str(self.step) + '.h5'
            # 更新保存数据的路径
            print('保存模型：' + self.save_weights_path)
        self.evaluate_net.save_weights(self.save_weights_path)
