<a name="6bc4ecf0"></a>
## 项目结构

- game_player 
   - __init__.py
   - brain.py
   - control_keyboard_keys.py
   - detect_keyboard_keys.py
   - grab_screen.py
   - others.py
   - run.py
   - video_recorder.py

<a name="e655a410"></a>
## 安装

<a name="889ea99a"></a>
#### 安装 Anaconda3

[https://www.anaconda.com/](https://www.anaconda.com/)

<a name="993df7fe"></a>
#### 创建虚拟环境和安装依赖

```shell
conda create -n game_AI python=3.8
conda activate game_AI
conda install pandas
conda install matplotlib
conda install pywin32
pip install opencv-python>=4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorflow>=2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c conda-forge jupyterlab
conda install ffmpeg -c conda-forge  # 安装 ffmpeg 用于自动录制游戏视频
```

<a name="c5868890"></a>
## 项目介绍

本项目是作者在五一假期学完吴恩达的《机器学习》课程后，以B站UP主[林亦LYi](https://space.bilibili.com/4401694?spm_id_from=333.337.0.0)、[蓝魔digital](https://space.bilibili.com/270844959?spm_id_from=333.337.search-card.all.click)、[遇上雨也笑笑](https://www.bilibili.com/video/BV1NK411w7Rp/?spm_id_from=333.337.search-card.all.click&vd_source=bf823389b65483590a312cedf448bec4)为灵感，第一次尝试自己搭建AI应用。在设计奖励系统时，借鉴了论文《[Mitigating Cowardice for Reinforcement Learning](https://ieee-cog.org/2022/assets/papers/paper_111.pdf) 》解决了怂货AI的困局，Double DQN来加快强化学习的效率

在本次项目版本中，通过使用OpenCV对游戏画面进行特殊读取，读取血条所在的画面，并进行颜色处理，把颜色降到一定程度后，将出血条以外的颜色去除，即可实现对游戏主角的血量信息的精确提取。 然后通过自定义奖励系统，根据血量大小来计算奖励。<br />提取拳皇游戏中自己和敌人的动作相对较为复杂，可能需要通过YOLO等计算机视觉神经网络来对游戏画面进行特征提取，提出后获得的数据会比较干净,训练效果可能更好。但由于作者的能力和精力均有限，因此采用的是将游戏画面传入卷积神经网络，由它自动提取游戏元素中的特征，并返回下一步玩家行动。<br />通过以上获得的信息可以定义一个经验结构如下：
```python
self.brain.replayer.store(
    observation,#当前游戏画面
    action,#采取的动作
    reward,#得到的奖励
    next_observation#下一个游戏画面
)

```

如果你有兴趣阅读源码可能看得到自动录制对战视频的代码注释，但我实现这个功能中遇到了一些困难，我的想法是弄一个自动把胜利对局的视频录制下来，但是使用opencv实现了一个demo，集成到我的AI代码中发现无法存游戏画面的数据，于是使用了ffmpeg做了一个可用的demo，集成到AI代码同样出现了无法保存游戏画面的数据，非常奇怪的问题，但时间有限只好作罢（做游戏精彩画面录制的时候还把好不容易搞起来的环境搞崩了emo一阵子，好在最后成功复原）<br />最后在训练数据可视化上面我做的并不是很好，我采用了matplotlib.pyplot来用把游戏训练的胜利数据导出来如下图：<br />![reward_10000.png](https://cdn.nlark.com/yuque/0/2023/png/29611082/1683123779857-7f141ce5-7cff-46e9-80cd-4d73d3cf59dd.png#averageHue=%23fcf8f7&clientId=u6b9fb3aa-6551-4&from=paste&height=600&id=u0d40c6c3&originHeight=600&originWidth=1200&originalType=binary&ratio=1&rotation=0&showTitle=false&size=48752&status=done&style=none&taskId=u60993040-5bab-42bb-9a4f-2e3751b27f1&title=&width=1200)<br />如图这个AI的胜率还是蛮高的（拳皇Wing-难度2），我自己亲自和AI对打如果不是利用AI不太擅长防御远程攻击的弱点，基本打不过它。
<a name="0f1bea9b"></a>

##  效果展示
https://www.bilibili.com/video/BV1Ro4y1w71h/
## 致谢

感谢B站UP主[林亦LYi](https://space.bilibili.com/4401694?spm_id_from=333.337.0.0)、[蓝魔digital](https://space.bilibili.com/270844959?spm_id_from=333.337.search-card.all.click)、[遇上雨也笑笑](https://www.bilibili.com/video/BV1NK411w7Rp/?spm_id_from=333.337.search-card.all.click&vd_source=bf823389b65483590a312cedf448bec4)等提供的灵感和帮助，感谢论文《[Mitigating Cowardice for Reinforcement Learning](https://ieee-cog.org/2022/assets/papers/paper_111.pdf) 》中相关的作者提供的思路
