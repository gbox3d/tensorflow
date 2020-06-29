# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np

import tensorflow as tf
from tensorflow import keras

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

# %%
tf.constant(100)


# %%
tensor_a = tf.constant(100) # 상수텐서 
print(tensor_a)
tensor_a.numpy() # 출력하기


# %%
tensor_b = tf.constant(3)
tensor_c = tf.constant(2)

tensor_d = tf.add(tensor_b,tensor_c)
print(tensor_d.numpy())


# %%
tensor_e = tf.multiply(tensor_b,tensor_c)
print(tensor_e.numpy())


# %%
tensor_f = tensor_b - tensor_c # 계산기호사용가능
print(tensor_f.numpy())


# %%
tensor_ma = tf.constant([[1,2],[3,4]])
tensor_mb = tf.constant([[2,0],[0,2]])
tensor_mc = tensor_ma * tensor_mb #벡터 곱하기
print(tensor_mc.numpy())
tensor_md = tf.matmul(tensor_ma,tensor_mb) #메트릭스 곱 
print(tensor_md.numpy())


# %%

tf.debugging.set_log_device_placement(True)

# 텐서 생성
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)



# %%
