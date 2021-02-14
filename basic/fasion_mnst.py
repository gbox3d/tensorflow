# %% 모듈준비
import tensorflow as tf
from tensorflow import keras
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
model = keras.Sequential([
    # 28*28 = 768 , Flatten 차원을 펼쳐주는 층
    keras.layers.Flatten(input_shape=(28, 28)),
    # Dense는 밀집연결층 ,밀집연결을 통해 차원을 변경한다.
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# %%
X_train, X_val, Y_train, Y_val = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42)

# %%
history = model.fit(X_train, Y_train, epochs=500,
          batch_size=512,
          validation_data=(X_val, Y_val), verbose=1
          )
# %%평가하기


# %% 예측값구하기
pred_label = model.predict(test_images)
# 결과값으로 나온 10개중 가장 높은 값의 인덱스 찾기
_pred_label = [pre.argmax() for pre in pred_label]

# %%
accuracy_score(test_labels, _pred_label)
# %%

cm = np.array(confusion_matrix(test_labels, _pred_label,
                               labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
sns.heatmap(cm, annot=True)
print(cm)

#%%
history_dict = history.history
history_dict.keys()
#%%
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
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
