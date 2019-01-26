#example of roi pooling and  using the spatial transformer network

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 

import tensorflow as tf

import matplotlib.pyplot as plt
import pdb
import sys
import numpy as np
from  PIL import Image

import spatial_transformer

im = Image.open("../P&C dataset/img/000000.jpg")
img_out = np.array(im)
plt.imshow(img_out)
plt.show()
img_out = img_out.astype(np.float32)

#theta corresponding to car bounding box
#7,15,62,104
xc= (7+62)/2.0
yc = (15+104)/2.0
theta_out= [62.0/128.0 , 0.0, (xc-64)/64.0, 0.0, 104/128.0, (yc-64)/64.0]




img = tf.constant(img_out)
img=tf.reshape(img, shape=[1, 128, 128, 3])
theta = tf.constant(theta_out)
theta= tf.reshape(theta, shape=[1, 6])


car_cropped = spatial_transformer.transformer(img, theta, [22,22])
car_cropped  = tf.reshape(car_cropped, shape = [22,22,3])

with tf.Session() as sess:
    car_cropped_out = sess.run(car_cropped)

car_cropped_out = car_cropped_out.astype(np.uint8)
plt.imshow(car_cropped_out)
plt.show()




















