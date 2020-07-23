#%%
import tensorflow as tf
import numpy as np

print( f'tensor versio : {tf.__version__}')

# from train_xor import create_model
_input = np.array([[0,0],[0,1],[1,0],[1,1]]).astype(np.float64)

#%%

def create_model() :
    layers = []
    layers.append(tf.keras.layers.Dense(2,activation=tf.nn.sigmoid))
    layers.append(tf.keras.layers.Dense(1,activation=tf.nn.sigmoid))
    model = tf.keras.Sequential(layers)
    print('layer setup')
    sgd = tf.keras.optimizers.SGD(lr=0.01,decay=0,momentum=0.99,nesterov=True)
    model.compile(optimizer=sgd,loss='mse',metrics=['mae','mse'])
    return model
    
model = create_model()
print(model(_input))

# %%
model.load_weights('training_1/xor.ckpt')

# %%

print(model(_input))

# %%
