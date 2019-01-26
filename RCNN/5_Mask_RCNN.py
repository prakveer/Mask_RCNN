import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 

import spatial_transformer

import tensorflow as tf

import utils
import matplotlib.pyplot as plt
import pdb
import sys
import numpy as np

def plotting(entry, namer):
        #plotting
    plt.plot(range(len(entry)), entry)
    plt.ylabel(namer)
    plt.xlabel('iterations')
    plt.show()
    return
  
train_rpn_clf_loss=[]
train_rpn_reg_loss=[]
train_rcnn_loss=[]
train_mask_loss =[]


class ConvNet(object):
    def __init__(self):
        self.lr = 0.01
        self.batch_size = 100
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.n_classes = 1
        self.skip_step = 2000 #intermediate prints, to enable reduce the skip step to less than 20 
        self.training=False
        self.Wa = 48
        self.seg_mask_size =21
        self.mask_train = tf.Variable(True, dtype= tf.bool, trainable = False)
        self.trainer = self.mask_train.assign(True)
        self.tester = self.mask_train.assign(False)






    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.get_pc_dataset(self.batch_size, self.seg_mask_size, option =1)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                                   train_data.output_shapes)
            self.img, self.binary_mask, self.ignore_mask, self.gt_regression, self.people_or_car, self.car_masks, self.people_masks, self.theta_car, self.theta_people= iterator.get_next()
            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)           # feature_dim = pool3.shape[1] * pool3.shape[2] * pool3.shape[3]
  

    def base(self):
        a1=utils.conv_bn_relu('conv1', self.img, filter=8, k_size=[3,3])
        pool1 = tf.layers.max_pooling2d(inputs=a1, 
                            pool_size=[2, 2], 
                            strides=2,
                            name='pool1')
###########################
        a2 =utils.conv_bn_relu('conv2',pool1, filter=16, k_size=[3,3])
        pool2 = tf.layers.max_pooling2d(inputs=a2, 
                                        pool_size=[2, 2], 
                                        strides=2,
                                        name='pool2')
##############################
        a3 =utils.conv_bn_relu('conv3',pool2, filter=32, k_size=[3,3])
        pool3 = tf.layers.max_pooling2d(inputs=a3, 
                                        pool_size=[2, 2], 
                                        strides=2,
                                        name='pool3')
##############################
        a4 =utils.conv_bn_relu('conv4',pool3, filter=64, k_size=[3,3])
        pool4 = tf.layers.max_pooling2d(inputs=a4, 
                                        pool_size=[2, 2], 
                                        strides=2,
                                        name='pool4')
###############################        
        self.base_network_output=utils.conv_bn_relu('conv5',pool4, filter=128, k_size=[3,3])

    def rpn(self):
        intermediate=utils.conv_bn_relu('rpn/conv1', self.base_network_output, filter=128, k_size=[3,3])
        

        # rpn classifier
        self.rpn_clf = tf.layers.conv2d(inputs=intermediate,
                                filters=1,
                                kernel_size=[1,1],
                                name='rpn/clf')
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.binary_mask, logits=self.rpn_clf)
        self.rpn_clf_loss = tf.reduce_mean(tf.multiply(entropy, self.ignore_mask), name='loss')
        self.rpn_clf_pred = tf.where(self.rpn_clf>0, tf.ones_like(self.rpn_clf), tf.zeros_like(self.rpn_clf))
        #accuracy ignoring the uncertain regions
        temp2 = 64-tf.reduce_sum(self.ignore_mask, axis=[1,2,3])
        #pdb.set_trace()
        temp = tf.equal(tf.multiply(self.binary_mask, self.ignore_mask), 
                                    tf.multiply(self.rpn_clf_pred, self.ignore_mask))
        self.rpn_clf_acc = tf.reduce_mean((tf.reduce_sum(tf.cast(temp, tf.float32), axis=[1,2,3])-temp2)/(64-temp2))

        #rpn regressor
        self.rpn_reg_pred= tf.layers.conv2d(inputs=intermediate,
                        filters=4,
                        kernel_size=[1,1],
                        name='rpn/reg', 
                        bias_initializer = tf.constant_initializer([64.0,64.0,128.0,128.0]))

        xa = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        Xa, Ya = tf.meshgrid(xa, xa)
        Xa= 16 * Xa+8
        Ya = 16 * Ya+8

        X, Y, W, H = tf.unstack(self.rpn_reg_pred, axis=3)
        tx = (X - Xa) / self.Wa
        ty = (Y- Ya) / self.Wa
        tw = tf.log(W/ self.Wa)
        th = tf.log(H/ self.Wa)

        tw2=tf.where(tf.is_nan(tw), tf.zeros_like(tw), tw)
        th2=tf.where(tf.is_nan(th), tf.zeros_like(th), th)
        tw2=tf.where(tf.is_inf(tw2), tf.zeros_like(tw2), tw2)
        th2=tf.where(tf.is_inf(th2), tf.zeros_like(th2), th2)

        t  = tf.stack([tx,ty,tw2,th2], axis =3)


        X_star, Y_star, W_star, H_star = tf.unstack(self.gt_regression, axis=3)
        tx_star = (X_star - Xa) / self.Wa
        ty_star = (Y_star- Ya) / self.Wa
        tw_star = tf.log(W_star/ self.Wa)
        th_star = tf.log(H_star/ self.Wa)

        tw_star2=tf.where(tf.is_nan(tw_star), tf.zeros_like(tw_star), tw_star)
        th_star2=tf.where(tf.is_nan(th_star), tf.zeros_like(th_star), th_star)
        tw_star2=tf.where(tf.is_inf(tw_star2), tf.zeros_like(tw_star), tw_star2)
        th_star2=tf.where(tf.is_inf(th_star2), tf.zeros_like(th_star), th_star2)

        t_star = tf.stack([tx_star,ty_star,tw_star2,th_star2], axis =3)

        differ = tf.abs(t-t_star)

        smooth_L1 =tf.where(differ>1, differ - 0.5, 0.5*differ*differ)


        smooth_L1 = tf.reduce_sum(smooth_L1, axis=3)
        smooth_L1=tf.reshape(smooth_L1, [self.batch_size, 8, 8, 1])

        L_reg =tf. multiply(smooth_L1, self.rpn_clf_pred)

        #print(smooth_L1, self.ignore_mask)
        self.rpn_reg_loss = (tf.reduce_sum(L_reg)/(4*tf.reduce_sum(self.rpn_clf_pred)))

        self.total_rpn_loss = self.rpn_clf_loss+ self.rpn_reg_loss

    


    def rcnn(self):        

        selected = tf.argmax(tf.reshape(self.rpn_clf, [self.batch_size, -1, 1]), axis=1)
      
        indices = tf.concat([tf.reshape(tf.range(self.batch_size, dtype = tf.int64), shape = selected.get_shape()), selected], 1)

        temp = tf.reshape(self.rpn_clf, [self.batch_size, -1, 1])
        temp_indices = tf.cast(tf.concat([indices, tf.zeros_like(selected)], 1), tf.int64)
        values =tf.squeeze(-10000 * tf.ones_like(selected, dtype = tf.float32)) 
        shape = [self.batch_size, 64, 1]
        delta = tf.SparseTensor(temp_indices, values, shape)
        temp2 = temp + tf.sparse_tensor_to_dense(delta)

        selected2 = tf.argmax(temp2, axis=1)      
        indices2 = tf.concat([tf.reshape(tf.range(self.batch_size, dtype = tf.int64), shape = selected2.get_shape()), selected2], 1)

        top2_indices = tf.concat([indices, indices2], axis = 0)

        gt_rcnn_clf = tf.reshape(tf.gather_nd(tf.reshape(self.people_or_car, [self.batch_size, -1]), top2_indices), [2*self.batch_size, 1])
        #print('gt_rcnn_clf',gt_rcnn_clf)


        selected_reg = tf.gather_nd(tf.reshape(self.rpn_reg_pred, [self.batch_size, -1, 4]), top2_indices)


        th0 = tf.reshape(selected_reg[:,2]/128.0, shape =[2*self.batch_size, 1])
        th2 = tf.reshape(selected_reg[:,0]/64.0 - 1.0, shape =[2*self.batch_size, 1])
        th4 = tf.reshape(selected_reg[:, 3]/128.0, shape =[2*self.batch_size, 1])
        th5 = tf.reshape(selected_reg[:,1]/64.0 - 1.0, shape =[2*self.batch_size, 1])

        self.theta_pred = tf.concat([th0, tf.zeros_like(th0), th2, tf.zeros_like(th0), th4, th5], axis = 1)

        self.input_to_transformer = tf.concat([self.base_network_output, self.base_network_output], axis= 0 )


        cropped_features = spatial_transformer.transformer(self.input_to_transformer, self.theta_pred, [4,4])
        cropped_features = tf.reshape(cropped_features, [2*self.batch_size, 4, 4, 128])
        rcnn_conv1=utils.conv_bn_relu('rcnn/conv1', cropped_features, filter=128, k_size=[3,3])
        rcnn_conv2=utils.conv_bn_relu('rcnn/conv2', rcnn_conv1, filter=128, k_size=[3,3])

        feature_dim = rcnn_conv2.shape[1] * rcnn_conv2.shape[2] * rcnn_conv2.shape[3]
        rcnn_conv2 = tf.reshape(rcnn_conv2, [2*self.batch_size, feature_dim])

        fc = tf.layers.dense(rcnn_conv2, 1, activation=None, name='rcnn/fc')
        #print('fc',fc)


        entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_rcnn_clf, logits=fc)

        self.rcnn_clf_loss = tf.reduce_mean(entropy)

        self.rcnn_clf_pred = tf.where(fc>0, tf.ones_like(fc), tf.zeros_like(fc))
        self.rcnn_clf_acc = tf.reduce_mean(tf.cast(tf.equal(gt_rcnn_clf, self.rcnn_clf_pred), tf.float32))
        self.total_loss = self.rcnn_clf_loss + self.total_rpn_loss
        #sys.exit()
        
    def  mask_rcnn(self):
        

        theta= tf.cond(self.mask_train, lambda: tf.cast(
        tf.concat([self.theta_car,  self.theta_people], axis =0), tf.float32), 
                       lambda: self.theta_pred)
        # theta = tf.concat([self.theta_car,  self.theta_people], axis =0)
        
        cropped_features = spatial_transformer.transformer(self.input_to_transformer, theta, [4,4])
        cropped_features = tf.reshape(cropped_features, [2*self.batch_size, 4, 4, 128])
        mask_conv1=utils.conv_bn_relu('mask/conv1', cropped_features, filter=128, k_size=[3,3])
        mask_conv2=utils.conv_bn_relu('mask/conv2', mask_conv1, filter=128, k_size=[3,3])
        
        final_conv = tf.layers.conv2d(inputs=mask_conv2,
                                filters=1,
                                kernel_size=[1,1],
                                name='mask/final_conv')
        scaled_masks1 = tf.layers.conv2d_transpose(final_conv, 1, [3,3], [2,2], activation=None)
        scaled_masks2 = tf.layers.conv2d_transpose(scaled_masks1, 1, [3,3], [2,2], activation=None)
        scaled_masks = tf.layers.conv2d_transpose(scaled_masks2, 1, [3,3], [1,1], activation=None)
        

        self.pred_masks = tf.where(scaled_masks>0, tf.ones_like(scaled_masks), tf.zeros_like(scaled_masks))
        
        self.gt_masks = tf.concat([self.car_masks, self.people_masks], axis =0)
        
        temp = tf.reshape(self.rcnn_clf_pred, shape=[2*self.batch_size, 1,1,1])
        temp2 = tf.cast(tf.broadcast_to(temp, shape=[2*self.batch_size, self.seg_mask_size, self.seg_mask_size, 1]) , tf.bool)
        masks_for_accuracy = tf.where(temp2, tf.concat([self.people_masks, self.people_masks], axis=0),
                                     tf.concat([self.car_masks, self.car_masks], axis=0))
        
        
        #print(masks_for_accuracy)
        
        self.seg_accuracy = tf.reduce_mean(tf.cast(tf.equal(masks_for_accuracy, self.pred_masks), tf.float32))
        
        #print(self.gt_masks, scaled_masks)
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gt_masks, logits=scaled_masks)
        self.seg_loss = tf.reduce_mean(entropy)   
        
           

        

   
    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        self.rpn_opt = tf.train.AdamOptimizer(self.lr).minimize(self.total_rpn_loss, 
                                                global_step=self.gstep)

        self.rcnn_opt = tf.train.AdamOptimizer(self.lr).minimize(self.rcnn_clf_loss, 
                                        global_step=self.gstep)

        self.total_opt = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss, 
                                        global_step=self.gstep)

        self.mask_opt = tf.train.AdamOptimizer(self.lr).minimize(self.seg_loss, 
                                        global_step=self.gstep)
        

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()
    

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.base()
        self.rpn()
        self.rcnn()
        self.mask_rcnn()
        self.optimize()
        #self.summary()




    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        sess.run(self.tester)        
        total_rpn_acc = 0
        total_rcnn_acc = 0
        total_mask_acc =0
        total_loss=0
        n_batches=0
        try:
            while True:
                rpn_a, rcnn_a, mask_a, imp = sess.run([self.rpn_clf_acc, self.rcnn_clf_acc, self.seg_accuracy, self.mask_train])

                #writer.add_summary(summaries, global_step=step)
                total_rpn_acc += rpn_a
                total_rcnn_acc += rcnn_a
                total_mask_acc += mask_a
                n_batches+=1

        except tf.errors.OutOfRangeError:
            pass

        print('Epoch: {0}, test accuracy rpn {1}, rcnn {2}, mask: {3}'.format(epoch, total_rpn_acc/n_batches, total_rcnn_acc/n_batches, total_mask_acc/n_batches))
        #test_acc.append(total_correct_preds/n_batches)
        #print('Took: {0} seconds'.format(time.time() - start_time))
        
    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        sess.run(self.trainer)
        total_loss = 0
        n_batches = 0
        total_rpn_loss = 0
        total_rcnn_loss =0
        total_rcnn_acc = 0
        total_mask_loss =0
        try:
            while True:
                _, rpn_clf_l, rpn_a, rpn_reg_l, rcnn_l, rcnn_a, car= sess.run([self.total_opt, self.rpn_clf_loss, 
                    self.rpn_clf_acc, self.rpn_reg_loss, self.rcnn_clf_loss, 
                    self.rcnn_clf_acc, self.car_masks])
                train_rpn_clf_loss.append(rpn_clf_l)
                train_rpn_reg_loss.append(rpn_reg_l)
                train_rcnn_loss.append(rcnn_l)
                
                _,mask_l, mask_a, gt, pred = sess.run([self.mask_opt, self.seg_loss, self.seg_accuracy, self.gt_masks, self.pred_masks])
                train_mask_loss.append(mask_l)
                #writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('step: {0}, rpn Loss clf: {1}, reg: {2}'.format(step, rpn_clf_l, rpn_reg_l))
                    print('mask loss: {0}, acc: {1}'.format(mask_l, mask_a))

                    #print('#####step: {0}, rcnn Loss clf: {1}, acc: ,{2}'.format(step, rcnn_l, rcnn_a))

                step += 1
                total_rpn_loss += rpn_clf_l+rpn_reg_l
                total_rcnn_loss += rcnn_l
                total_mask_loss += mask_l
                total_rcnn_acc += rcnn_a
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        #saver.save(sess, 'checkpoints/convnet_layers/cifar-convnet1', step)
        print('Epoch {0}, train loss rpn: {1}, rcnn: {2} mask: {3}'.format(epoch, total_rpn_loss/n_batches, total_rcnn_loss/n_batches, total_mask_loss/n_batches))
        #print('Took: {0} seconds'.format(time.time() - start_time))
        #print('Epoch {0}, train acc rcnn: {1}'.format(epoch, total_rcnn_acc/n_batches))
        return step


    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/convnet_layers')
        writer = tf.summary.FileWriter('./graphs/convnet_layers', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_layers/cifar-convnet1'))
            # if ckpt and ckpt.model_checkpoint_path:
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
                #if(test_acc[epoch] <= test_acc[epoch-1] and epoch!=0):
                #    print(epoch)
                #    break;

                
        writer.close()
        
#     def train_one_epoch_mask_branch():
#        with tf.Session() as sess:
          
      
    

if __name__ == '__main__':
    tf.reset_default_graph()
    model = ConvNet()
    model.build()
    model.train(n_epochs=100)
    plotting(train_rpn_clf_loss, 'train rpn clf loss') 
    plotting(train_rpn_reg_loss, 'train rpn reg loss') 
    plotting(train_rcnn_loss, 'train rcnn loss') 
    plotting(train_mask_loss, 'train mask loss') 
