#%%
import os
import tensorflow as tf
import numpy as np

TFLITE_PATH = "./my_yolov5s.tflite"

# example_input = get_numpy_example()
print(f"Using tensorflow {tf.__version__}") # make sure it's the nightly build
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#%%
interpreter.set_tensor(input_details[0]['index'], example_input)
interpreter.invoke()
print(interpreter.get_tensor(output_details[0]['index'])) # printing the result
# %%
