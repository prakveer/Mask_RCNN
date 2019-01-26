import os
import struct
import urllib
import pickle
import pdb 
from PIL import Image

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import cv2


tf.set_random_seed(1)
train_fraction= 0.8
img_dir = '../P&C dataset/img'
car_mask_dir = '../P&C dataset/mask/car'
people_mask_dir = '../P&C dataset/mask/people'
label_car_file= '../P&C dataset/label_car.txt'
label_people_file='../P&C dataset/label_people.txt'
im_height =128
im_width = 128
im_channels = 3
iou_threshold = 0.4 # adjusted from 0.5 to get more proposals


def file_parser(name, length):
    l =0
    px=np.zeros(length, dtype=np.int32)
    py=np.zeros(length, dtype=np.int32)
    pw=np.zeros(length, dtype=np.int32)
    ph=np.zeros(length, dtype=np.int32)
    pcx= np.zeros(length, dtype=np.int32)
    pcy=np.zeros(length, dtype=np.int32)    

    with open(name) as f:
        for line in f:
            #pdb.set_trace()
            t1,t2,t3,t4 = line.split(',')
            px[l]=int(t1)
            py[l]=int(t2)
            pw[l]=int(t3)
            ph[l]=int(t4)
            #print(px[l],py[l],pw[l],ph[l])
            #pdb.set_trace()
            pcx[l] = (px[l]+pw[l]/2)
            pcy[l] = (py[l]+ph[l]/2)

            l=l+1

    return px,py,pw,ph, pcx, pcy


def calc_iou_masks(bbox_object_out, pcx, pcy, pw, ph):

    dataset_size=len(pcx)
    bbox_object = tf.constant(bbox_object_out, dtype=tf.float32)
    objectsum = (tf.reduce_sum(bbox_object, axis=[1,2,3]))/256;

    bbox_object = tf.layers.max_pooling2d(inputs=bbox_object, 
                                        pool_size=[8, 8], 
                                        strides=16,
                                        padding='SAME')
    objectsum = tf.reshape(objectsum, [-1, 1,1,1])
    kernel=tf.constant(shape=[3,3, 1, 1], value=1, dtype=tf.float32)

    # kernel=tf.constant(shape=[48,48, 1, 1], value=1, dtype=tf.float32)
    #             tf.layers.max_pooling2d(inputs=intersection_with_object, 
    #                                     pool_size=[48, 48], 
    #                                     strides=16,
    #                                     padding='SAME')



    intersection_with_object = tf.nn.conv2d(bbox_object, kernel, strides=[1,1,1,1], padding='SAME')
    #print(intersection_with_object_temp)
    # intersection_with_object = tf.layers.max_pooling2d(inputs=intersection_with_object, 
    #                                     pool_size=[48, 48], 
    #                                     strides=16,
    #                                     padding='SAME')

    temp = 1*tf.ones(shape = [dataset_size,8,8, 1])
    union_with_object= tf.nn.conv2d(temp, kernel, strides=[1,1,1,1], padding='SAME')+objectsum - intersection_with_object
    #pdb.set_trace()
    # with tf.Session() as sess:
    #     u_out, inter_out, bbox_out =sess.run([union_with_object, intersection_with_object, bbox_object])

    # #pdb.set_trace()
    # plt.imshow(np.squeeze(u_out[61,:,:,:]))
    # plt.show()

    # plt.imshow(np.squeeze(inter_out[61,:,:,:]))
    # plt.show()

    # plt.imshow(np.squeeze(bbox_object_out[61,:,:,:]))
    # plt.show()



    iou_object= tf.divide(intersection_with_object, union_with_object)

    binary_object = tf.where(iou_object>iou_threshold,tf.ones_like(iou_object), iou_object)
    binary_object = tf.where(iou_object<0.1,tf.zeros_like(iou_object), binary_object)
    regression_object = tf.zeros(shape=[dataset_size, 8, 8, 4])
    p1 = tf.reshape(tf.cast(pcx, tf.float32), [-1,1,1,1])
    p2 = tf.reshape(tf.cast(pcy, tf.float32), [-1,1,1,1])
    p3 = tf.reshape(tf.cast(pw, tf.float32), [-1,1,1,1])
    p4 = tf.reshape(tf.cast(ph, tf.float32), [-1,1,1,1])


    pa =tf.where(tf.equal(binary_object,tf.ones_like(binary_object)), 
        tf.multiply(p1, tf.ones_like(binary_object)), 
        tf.zeros_like(binary_object) )
    pb=tf.where(tf.equal(binary_object,tf.ones_like(binary_object)), 
        tf.multiply(p2, tf.ones_like(binary_object)), 
        tf.zeros_like(binary_object) )
    pc=tf.where(tf.equal(binary_object,tf.ones_like(binary_object)), 
        tf.multiply(p3, tf.ones_like(binary_object)), 
        tf.zeros_like(binary_object) )
    pd=tf.where(tf.equal(binary_object,tf.ones_like(binary_object)), 
        tf.multiply(p4, tf.ones_like(binary_object)), 
        tf.zeros_like(binary_object) )
    
    return iou_object, binary_object, pa, pb, pc, pd
    #, intersection_with_people, sum_with_people, bbox_people




def pc_data_parser(seg_mask_size = 22):
    images=[]
    labels=[]
    files = os.listdir(img_dir)
    files = sorted(files)
    print("Total Number of images: ", len(files))
    imgs =np.zeros(shape=[len(files), im_height, im_width, im_channels], dtype=np.float32)
    
    people_masks_out =np.zeros(shape=[len(files), seg_mask_size, seg_mask_size, 1], dtype=np.float32)
    car_masks_out = np.zeros(shape=[len(files), seg_mask_size, seg_mask_size, 1], dtype=np.float32)
    
    maskfiles = os.listdir(car_mask_dir)
    maskfiles = sorted(maskfiles)

    i=0
    for file in files:
        im=Image.open(img_dir +'/'+ file)
        imgs[i,:,:,:]=np.array(im, dtype=np.float32)
        #pdb.set_trace()
        i +=1
        
    i=0    
    for file in maskfiles:

        c_mask = cv2.imread(car_mask_dir +'/'+ file )
        car_masks_out[i,:,:,0] = cv2.resize(cv2.cvtColor(c_mask, cv2.COLOR_BGR2GRAY), (seg_mask_size,seg_mask_size))

        p_mask = cv2.imread(people_mask_dir +'/'+ file )
        people_masks_out[i,:,:,0] = cv2.resize(cv2.cvtColor(p_mask, cv2.COLOR_BGR2GRAY), (seg_mask_size,seg_mask_size))
        i+=1

    bbox_people_out =np.zeros(shape=[len(files), im_height, im_width, 1], dtype=np.float32)
    bbox_car_out =np.zeros(shape=[len(files), im_height, im_width, 1], dtype=np.float32)

    #iou =np.zeros(shape=[len(files), im_height, im_width, 1], dtype=np.float32)
    dataset_size = len(files)

    px,py,pw,ph, pcx, pcy=file_parser(label_people_file, len(files))
    cx,cy,cw,ch, ccx, ccy=file_parser(label_car_file, len(files))
    for i in range(len(files)):
        bbox_people_out[i, py[i]:py[i]+ph[i],px[i]:px[i]+pw[i]+1,:] = 1
        bbox_car_out[i, cy[i]:cy[i]+ch[i],cx[i]:cx[i]+cw[i]+1,:] = 1

    iou_people, binary_people, pa, pb, pc, pd= calc_iou_masks(bbox_people_out, pcx, pcy, pw, ph)
    #, inter, su, bb
    iou_car, binary_car, ca, cb, cc, cd= calc_iou_masks(bbox_car_out, ccx, ccy, cw, ch)

    #FOR MASK RCNN TRAINING

    theta_people_out = np.array([pw/128.0, np.zeros_like(pw), pcx/64.0-1.0, np.zeros_like(pw), ph/128.0, pcy/64.0 -1.0])
    theta_people_out = np.transpose(theta_people_out)

    theta_car_out = np.array([cw/128.0, np.zeros_like(cw), ccx/64.0-1.0, np.zeros_like(cw), ch/128.0, ccy/64.0 -1.0])
    theta_car_out = np.transpose(theta_car_out)


    # with tf.Session() as sess:
    #     iou_people_out, binary_people_out, pa_out, inter_out, su_out, bb_out= sess.run([iou_people, binary_people, pa, inter, su, bb])

    # print(py[1], px[1])
    # i=1999
    # plt.imshow(np.squeeze(bbox_car_out[i,:,:,:]))
    # plt.show()

    # plt.imshow(np.squeeze(bbox_people_out[i,:,:,:]))
    # plt.show()


    # plt.imshow(np.squeeze(imgs[1,:,:,:]))
    # plt.show()
    #print(su_out[3])

    # plt.imshow(np.squeeze(bb_out[3,:,:,:]))
    # plt.show()

    # plt.imshow(np.squeeze(inter_out[3,:,:,:]))
    # plt.show()

    # plt.imshow(np.squeeze(binary_people_out[1,:,:,:]))
    # plt.show()

    # plt.imshow(np.squeeze(pa_out[1,:,:,:]))
    # plt.show()



    

    binary_mask = tf.where(iou_people>iou_car, binary_people, binary_car)
    gt_regression_1 = tf.where(iou_people>iou_car, pa, ca)
    gt_regression_2 = tf.where(iou_people>iou_car, pb, cb)
    gt_regression_3 = tf.where(iou_people>iou_car, pc, cc)
    gt_regression_4 = tf.where(iou_people>iou_car, pd, cd)

    people_or_car = tf. where(iou_people>iou_car, tf.ones_like(iou_people), tf.zeros_like(iou_car))

    gt_regression = tf.squeeze(tf.stack([gt_regression_1,gt_regression_2,gt_regression_3,gt_regression_4], axis=3))
    ignore_mask = tf.where(tf.logical_or(tf.equal(binary_mask,tf.ones_like(binary_people)), tf.equal(binary_mask,tf.zeros_like(binary_people))), tf.ones_like(binary_mask), tf.zeros_like(binary_mask) )

    # with tf.Session() as sess:
    #     bi_out, gt_out =sess.run( [binary_mask, gt_regression])

    # for i in range(2000):
    #     if(np.amax(bi_out[i,:,:,:])<1.0):
    #         pdb.set_trace()

    # plt.imshow(np.squeeze(ig_out[i,:,:,:]))
    # plt.show()

    # plt.imshow(np.squeeze(bi_out[i,:,:,:]))
    # plt.show()

    # plt.imshow(np.squeeze(gt_out[i,:,:,1]))
    # plt.show()

    im_tensor = tf.constant(imgs)
    car_masks = tf.constant(car_masks_out)
    people_masks = tf.constant(people_masks_out)
    theta_car = tf.constant(theta_car_out)
    theta_people = tf.constant(theta_people_out)

    return im_tensor, binary_mask, ignore_mask, gt_regression, len(files), people_or_car, car_masks, people_masks, theta_car, theta_people
    #return

def conv_bn_relu(scope_name, input, filter, k_size, stride=(1,1), padding='SAME'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        conv = tf.layers.conv2d(inputs=input,
                                filters=filter,
                                kernel_size=k_size,
                                strides=stride,
                                padding=padding)
        batch_norm=tf.layers.batch_normalization(inputs=conv, training=True)
        a =tf.nn.relu(batch_norm, name=scope.name)

    return a



def get_pc_dataset(batch_size, seg_mask_size = 22, option = 0):

    im_tensor, binary_mask, ignore_mask, gt_regression,  dataset_size, people_or_car, car_masks, people_masks, theta_car, theta_people  = pc_data_parser(seg_mask_size)

    # Step 2: Create datasets and iterator
    if(option == 1):
        data = tf.data.Dataset.from_tensor_slices((im_tensor, binary_mask, ignore_mask, gt_regression, people_or_car, car_masks,people_masks, theta_car, theta_people))
    elif(option ==0):
        data = tf.data.Dataset.from_tensor_slices((im_tensor, binary_mask, ignore_mask, gt_regression, people_or_car))
    else:
        print("WRONG OPTION SEE get_pc_dataset FUNCTION IN utils.py")
        sys.exit()
    #data = data.shuffle(dataset_size)

    train_data = data.take(int(train_fraction* dataset_size))
    test_data = data.skip(int(train_fraction* dataset_size))
    #test_data = test_data.shuffle(int(train_fraction* dataset_size))

    train_data = train_data.batch(batch_size)
    test_data = test_data.batch(batch_size)

    return train_data, test_data


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


#tf.reset_default_graph()
#tr, test = get_pc_dataset(100, option=1)