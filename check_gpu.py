#%%
import tensorflow as tf


print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

# %%
tf.config.list_physical_devices('GPU')

# %%
tf.test.is_gpu_available()

# %%
tf.test.is_built_with_cuda()

# %%
