import tensorflow as tf

# 输出 TensorFlow 版本号和是否使用 GPU 进行计算
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.test.is_gpu_available())