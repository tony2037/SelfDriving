# reference : https://github.com/bokorn/Keras-and-Theano-layers-for-Switched-Pooling/blob/master/test_layers.py
from keras.layers import Input, merge, Dense
from keras.models import Model
from invertible_layers import MaxPoolSwitch2D, UnPoolSwitch2D, max_only_lambda, switch_only_lambda
import cv2
import numpy as np
import matplotlib.pyplot as plt

index_type='flattened'
index_scope='global'        

use_color = False

if(use_color):
    input_shape = (None, 3,256,256)
else:
    input_shape = (None, 1,256,256)

inputs = Input(shape=(input_shape[1:]))

max_switch_1 = MaxPoolSwitch2D((2, 2), strides=(2,2), 
                               index_type=index_type, 
                               index_scope=index_scope,
                               name="max_switch_1")(inputs)
                               
switch_1 = switch_only_lambda(name="switch_1", 
                              index_type=index_type)(max_switch_1)    
max_1 = max_only_lambda(name="max_1", 
                        index_type=index_type)(max_switch_1)

merged_max_switch_1 = merge([max_1, switch_1], 
                            mode='concat',
                            concat_axis=1,
                            name="merged_max_switch_1")
                            
unpool_1 = UnPoolSwitch2D((2, 2), strides=(2,2), 
                          index_type=index_type, 
                          index_scope=index_scope,
                          original_input_shape = (None, 3,256,256),
                          name="unpool_1")(merged_max_switch_1)

model = Model(input=inputs, output=[unpool_1, max_1])

# Parrot image from ImageNet http://image-net.org/
img_in = cv2.imread('./dataset/preprocess_image/x/bremen_000002_000019_leftImg8bit.png')
if(not use_color):
    img_in = np.expand_dims(cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY), axis=2)
    plt.gray()
    
img_batch = np.stack([img_in.astype('float32').transpose((2, 0, 1))], axis=0)
outputs = model.predict(img_batch)
img_out = outputs[0][0]
img_out = img_out.astype('uint8').transpose((1, 2, 0))
img_pool = outputs[1][0]
img_pool = img_pool.astype('uint8').transpose((1, 2, 0))

plt.subplot(131)
plt.imshow(np.squeeze(img_in))
plt.subplot(132)
plt.imshow(np.squeeze(img_pool))
plt.subplot(133)
plt.imshow(np.squeeze(img_out))
plt.show()
