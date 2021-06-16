#%%
import os
# import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util

from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from time import time

from tflite_runtime.interpreter import Interpreter
print(f'load tflite ok')
#%%
PATH_TO_CKPT = './yolov5s-fp16.tflite'
# PATH_TO_CKPT = './best-fp16.tflite'

#%%
# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()
# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
        
print(input_details)
print(output_details)

#%%

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

print(input_details[0]['shape'])

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

print(f'{width},{height}')
# %%
min_conf_threshold = 0.5
image_path = '../res/bus.jpg'
# Load image and resize to expected shape [1xHxWx3]
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imH, imW, _ = image.shape 
image_resized = cv2.resize(image_rgb, (width, height))
print(image_resized.shape)
input_data = np.expand_dims(image_resized, axis=0)
# print(input_data.shape)
# input_data = np.einsum('klij->kjli',input_data)
print(input_data.shape)

#%%

_startTick = time()

# Normalize pixel values if using a floating model (i.e. if model is non-quantized)
if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

# Perform the actual detection by running the model with the image as input
interpreter.set_tensor(input_details[0]['index'],input_data)
interpreter.invoke()
pred = interpreter.get_tensor(output_details[0]['index'])
print(pred)
pred[..., :4] *= 640


print(f'invoke time :  { time() - _startTick }')

# %%
