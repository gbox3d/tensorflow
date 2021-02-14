# %% 모듈준비
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Input, Flatten

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices(
    "GPU") else "사용 불가능")

# %%
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

print(f'train_image sahpe : {train_images.shape}')
print(f'train_label sahpe : {train_labels.shape}')
print(f'train_image sahpe : {test_images.shape}')
# %% 정규화
train_images = train_images / 255.0
test_images = test_images / 255.0

# %%
# 768->128->10
# model = keras.Sequential([
#     # 28*28 = 768 , Flatten 차원을 펼쳐주는 층
#     keras.layers.Flatten(input_shape=(28, 28)),
#     # Dense는 밀집연결층 ,밀집연결을 통해 차원을 변경한다.
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()

# lenet-5
model = Sequential()
model.add(layers.Conv2D(6, (3, 3), padding='same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(16, (2, 2), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])


print('compile ok')


# %%
_train_images = np.expand_dims(train_images, axis=3)
_train_labels = np.expand_dims(train_labels, axis=1)
X_train, X_val, Y_train, Y_val = train_test_split(
    _train_images, _train_labels, test_size=0.1, random_state=42)

X_train.shape

# %% training
history = model.fit(X_train, Y_train, epochs=100,
                    batch_size=512,
                    validation_data=(X_val, Y_val),
                    verbose=1
                    )
# %% 예측값구하기
x_test = np.expand_dims(test_images, axis=3)
pred_label = model.predict(x_test)
# 결과값으로 나온 10개중 가장 높은 값의 인덱스 찾기
_pred_label = [pre.argmax() for pre in pred_label]

# %%
accuracy_score(test_labels, _pred_label)
# %%

cm = np.array(confusion_matrix(test_labels, _pred_label,
                               labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
sns.heatmap(cm, annot=True)
print(cm)

# %%
history_dict = history.history
history_dict.keys()
# %%
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo"는 "파란색 점"입니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# %%
