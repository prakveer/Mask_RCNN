#Mask_RCNN RPN - regressor analysis 
#disabled summary and checkpoint save/restore

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 

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


train_acc=[]
test_acc=[]
train_loss=[]
test_loss=[]


class ConvNet(object):
    def __init__(self):
        self.lr = 0.01
        self.batch_size = 100
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.n_classes = 2
        self.skip_step = 200  #intermediate prints, to enable reduce the skip step to less than 20 
        self.training=False
        self.Wa = 48


    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.get_pc_dataset(self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                                   train_data.output_shapes)
            self.img, self.binary_mask, self.ignore_mask, self.gt_regression= iterator.get_next()

            self.train_init = iterator.make_initializer(train_data)  
            self.test_init = iterator.make_initializer(test_data)           
  

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
        rpn_clf = tf.layers.conv2d(inputs=intermediate,
                                filters=1,
                                kernel_size=[1,1],
                                name='rpn/clf')
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.binary_mask, logits=rpn_clf)
        self.rpn_clf_loss = tf.reduce_mean(tf.multiply(entropy, self.ignore_mask), name='loss')
        self.rpn_clf_pred = tf.where(rpn_clf>0, tf.ones_like(rpn_clf), tf.zeros_like(rpn_clf))
        #accuracy ignoring the uncertain regions
        temp2 = 64-tf.reduce_sum(self.ignore_mask, axis=[1,2,3])
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
        tx = (X ) / self.Wa
        ty = (Y) / self.Wa
        tw = tf.log(W/ self.Wa + 1e-7)
        th = tf.log(H/ self.Wa + 1e-7)


        t  = tf.stack([tx,ty,tw,th], axis =3)
        t=tf.where(tf.is_nan(t), tf.zeros_like(t), t)
        t=tf.where(tf.is_inf(t), tf.zeros_like(t), t)


        X_star, Y_star, W_star, H_star = tf.unstack(self.gt_regression, axis=3)
        tx_star = (X_star ) / self.Wa
        ty_star = (Y_star) / self.Wa
        tw_star = tf.log(W_star/ self.Wa+ 1e-7)
        th_star = tf.log(H_star/ self.Wa+ 1e-7)

        t_star = tf.stack([tx_star,ty_star,tw_star,th_star], axis =3)
        
        t_star=tf.where(tf.is_nan(t_star), tf.zeros_like(t_star), t_star)
        t_star=tf.where(tf.is_inf(t_star), tf.zeros_like(t_star), t_star)

        differ = tf.abs(t-t_star)     
        
 

        smooth_L1 =tf.where(differ>1, differ - 0.5, 0.5*differ*differ)
        smooth_L1 = tf.reduce_sum(smooth_L1, axis=3)
        smooth_L1=tf.reshape(smooth_L1, [self.batch_size, 8, 8, 1])


        self.binary = tf.where(tf.equal(self.binary_mask, tf.ones_like(self.binary_mask)),
            tf.ones_like(self.binary_mask), tf.zeros_like(self.binary_mask) )
        L_reg =tf. multiply(smooth_L1, self.binary)


        #print(smooth_L1, self.ignore_mask)
        self.L_reg = (tf.reduce_sum(L_reg)/(4*tf.reduce_sum(self.binary)))
        self.total_rpn_loss = 0.5*self.rpn_clf_loss+ self.L_reg
   
    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        self.rpn_opt = tf.train.AdamOptimizer(self.lr).minimize(self.total_rpn_loss, 
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
        self.optimize()       
        
        #self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        self.training = True
        total_reg_loss = 0.0
        n_batches = 0
        try:
            while True:
                _, l, acc, reg_l, reg_out, gt_out, binary_out = sess.run([self.rpn_opt, self.rpn_clf_loss, 
                    self.rpn_clf_acc, self.L_reg, self.rpn_reg_pred,self.gt_regression, self.binary])
                train_loss.append(reg_l)
                train_acc.append(acc)
                #writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: clf: {1}, reg: {2}'.format(step, l, reg_l))

                step += 1
                total_reg_loss += reg_l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        #saver.save(sess, 'checkpoints/convnet_layers/cifar-convnet1', step)
        
        ###########################################################################
        ##PRINTING THE REGRESSION O/P MASKED AND THE GT REGRESSION FOR COMPARISION
        print('REGRESSION O/P MASKED (cast to int) & GT REGRESSION')
        final_pred_show = reg_out*binary_out
        final_pred_show = final_pred_show.astype(int)
        print( final_pred_show[0,:,:,0])
        print(gt_out[0,:,:,0])
        print('train regression loss at epoch {0}: {1}'.format(epoch, total_reg_loss/n_batches))
        #print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0.0
        total_reg_loss=0.0
        n_batches=0
        try:
            while True:
                accuracy_batch, rpn_reg_loss = sess.run([self.rpn_clf_acc, self.L_reg])

                #writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
                total_reg_loss += rpn_reg_loss
                n_batches+=1

        except tf.errors.OutOfRangeError:
            pass

        print('Test clf Accuracy at epoch {0}: {1}, reg loss: {2} '.format(
            epoch,  total_correct_preds/n_batches, total_reg_loss/n_batches))
        test_acc.append(total_correct_preds/n_batches)
        test_loss.append(total_reg_loss/n_batches)
        #print('Took: {0} seconds'.format(time.time() - start_time))

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


                
        writer.close()

if __name__ == '__main__':
    tf.reset_default_graph()
    model = ConvNet()
    model.build()
    model.train(n_epochs=40) 
    plotting(train_loss, 'train regression loss') 
    plotting(train_acc, 'train clf accuracy') 
    plotting(test_acc, 'test clf accuracy') 
    plotting(test_loss, 'test regression loss')
