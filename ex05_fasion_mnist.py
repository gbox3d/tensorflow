#%%
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

#%% keras.datasets.mnist
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images[0],train_labels[0])

#%% 
print(train_images.shape)
# %%
train_images = train_images / 255.0

test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
# %%
#층설계
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.summary()
# 모델컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print('model compiled')

#%%훈련 
model.fit(train_images, train_labels, epochs=50)
# %%
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)
# %% 예측하기 
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
# %% 단일 이미지 예측 하기 
img = test_images[0]
print(img.shape)
predictions_single = model.predict(np.expand_dims(img,0))
print(predictions_single)

#%%
plt.grid(False)
#plt.xticks([])
plt.xticks(range(10), class_names, rotation=45)
plt.yticks([])
thisplot = plt.bar(range(10), predictions_single[0], color="#777777")
plt.ylim([0, 1])
predicted_label = np.argmax(predictions_single)
thisplot[predicted_label].set_color('red')
#%%