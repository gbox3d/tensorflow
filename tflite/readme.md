
## convert to tfLite

* onnx -> pb  
필수 모듈
```sh
pip install tensorflow==2.5.0
pip install tensorflow-addons==0.13.0
pip install onnx-tf==1.8.0
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
