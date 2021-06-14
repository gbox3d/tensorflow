
## convert to tfLite

* onnx -> pb  
필수 모듈
```sh
pip install tensorflow 
pip install tensorflow-addons
pip install onnx-tf 
```

convert_tf.py 소스 참고


* pb -> tflite
```sh
tflite_convert \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```


## 참고 자료 :
https://www.tensorflow.org/lite/convert
