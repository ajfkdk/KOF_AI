import re
from pathlib import Path

from game_player.run import Agent

folder_path = r"D:\data\训练文件"
vedio_folder= r"D:\data\vedio\\"
path=None
max_number=0
# 使用正则表达式从文件名中提取数字部分，并找到最大的数字
pattern = r"train_(\d+)\.h5"
numbers = []
for file_path in Path(folder_path).glob("train_*.h5"):
    file_name = file_path.name  # 获取文件名
    match_result = re.match(pattern, file_name)
    if match_result is not None:
        numbers.append(int(match_result.group(1)))

if len(numbers) == 0:
    print("文件夹中没有符合要求的文件")


else:
    max_number = max(numbers)
    # 构造目标文件名
    target = f"train_{max_number}.h5"
    print(f"目标文件名为：{target}")
    path = folder_path + '\\' + target

agent = Agent(
    # save_memory_path = target + '_memory.json',    # 注释这行就不保存记忆
    # load_memory_path = target + '_memory.json',    # 注释这行就不加载记忆
    save_weights_path=path,  # 注释这行就不保存模型权重
    load_weights_path=path ,  # 注释这行就不加载模型权重
    max_steps=max_number,
    folder_path=folder_path,
    vedio_folder=vedio_folder
)
agent.run()
