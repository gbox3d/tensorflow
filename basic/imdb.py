#%%
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import matplotlib._qhull

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

# %% util functions

def drawHistory(history) :
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

#%% dataset download
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("dataset ready")

#%% 단어 사전 만들기 
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()} # 4개단어추가 (0~3)
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

#%% 단어수를 256 단어로 만들기 빈공간 <pad>로 체우기
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# %%
train_data.shape
# %% 입력 크기는 영화 리뷰 데이터셋에 적용된 어휘 사전의 크기입니다(10,000개의 단어)
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
# %%
model.compile(optimizer='adam',
              loss='binary_crossentropy', #이진분류문제 0,1 일경우 쓰는 손실함수
              metrics=['accuracy'])

#%%
#검증세트 만들기 
# 검증세트와 테스트 테스의 차이점 : 검증세트는 훈련중에 피드백을 위한 데이터이다. 반면 테스트세트는 훈련을 마치고 성능 평가할때 쓰는데이터이다.
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# %%
def test_no_early_stop() :
    model.save_weights('init.h5')

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=100,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)
    
    results = model.evaluate(test_data,  test_labels, verbose=2)
    print(results)

    pred = model.predict(test_data)

    _pred = [ int(round(v[0])) for v in pred]

    accuracy_score(test_labels,_pred)

    drawHistory(history)
    model.load_weights('init.h5')


#%%조기 종료 , 지정된 에포크동안 성능 향상이 없으면 종료 시킴 (오버 피팅방지 )
# patience 매개변수는 성능 향상을 체크할 에포크 횟수입니다
def test_early_stop() :
    model.save_weights('init.h5')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=100,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1,
                        callbacks=[early_stop]
                        )
    results = model.evaluate(test_data,  test_labels, verbose=2)
    print(results)

    pred = model.predict(test_data)

    _pred = [ int(round(v[0])) for v in pred]

    accuracy_score(test_labels,_pred)
    
    drawHistory(history)

    model.load_weights('init.h5')


# %%
test_no_early_stop()
test_early_stop()