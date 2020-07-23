#%%
import tensorflow as tf
import numpy as np

print( f'tensor versio : {tf.__version__}')

#%%
# https://m.blog.naver.com/PostView.nhn?blogId=atelierjpro&logNo=221595798266&proxyReferer=https:%2F%2Fwww.google.com%2F

_input = np.array([[0,0],[0,1],[1,0],[1,1]]).astype(np.float64)
_output = np.array([[0],[1],[1],[0]]).astype(np.float64)

train_dataset = tf.data.Dataset.from_tensor_slices((_input,_output))
test_dataset = tf.data.Dataset.from_tensor_slices((_input,_output))
BATCH_SIZE=1
SHUFFLE_BUFFER_SIZE = 4

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

print("data setup")

def create_model() :
    layers = []
    layers.append(tf.keras.layers.Dense(2,activation=tf.nn.sigmoid))
    layers.append(tf.keras.layers.Dense(1,activation=tf.nn.sigmoid))
    model = tf.keras.Sequential(layers)
    print('layer setup')
    sgd = tf.keras.optimizers.SGD(lr=0.01,decay=0,momentum=0.99,nesterov=True)
    model.compile(optimizer=sgd,loss='mse',metrics=['mae','mse'])
    return model



#%%
if __name__ == "__main__" :
    model = create_model()
    # model.summary()
    model.fit(train_dataset,epochs=500,
        validation_data = test_dataset
        )
    model.save_weights("training_1/xor.ckpt")
    print(model(_input))



# %%
