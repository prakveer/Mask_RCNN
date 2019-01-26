#Region proposal network classifier part of Mask-RCNN
#disabled summary writer and checkpoint save/restore

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 

import tensorflow as tf
import utils
import matplotlib.pyplot as plt
import pdb
import sys
import numpy as np


#plotting
def plotting(entry, namer):
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
        self.skip_step = 200  #intermediate prints
        self.training=False
        self.Wa = 48


    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.get_pc_dataset(self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                                   train_data.output_shapes)
            self.img, self.binary_mask, self.ignore_mask, self.gt_regression= iterator.get_next()

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
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

    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        self.rpn_opt = tf.train.AdamOptimizer(self.lr).minimize(self.rpn_clf_loss, 
                                                global_step=self.gstep)

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()
    
    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), self.label)
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.base()
        self.rpn()
        self.optimize()
        #self.eval()
        
        #disabled summary writer 
        #self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        self.training = True
        total_loss = 0.0
        n_batches = 0
        try:
            while True:
                _, l, acc = sess.run([self.rpn_opt, self.rpn_clf_loss, 
                    self.rpn_clf_acc])
                
                #storing losses for plotting
                train_loss.append(l)
                train_acc.append(acc)
                #writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}, acc: {2}'.format(step, l, acc))

                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        #saver.save(sess, 'checkpoints/convnet_layers/cifar-convnet1', step)        
        print('train loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
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
                accuracy_batch = sess.run(self.rpn_clf_acc)

                #writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
                n_batches+=1

        except tf.errors.OutOfRangeError:
            pass

        print('Test clf Accuracy at epoch {0}: {1} '.format(
            epoch,  total_correct_preds/n_batches))
        test_acc.append(total_correct_preds/n_batches)
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
    model.train(n_epochs=20) 
    plotting(train_loss, 'train clf loss') 
    plotting(train_acc, 'train clf accuracy') 
    plotting(test_acc, 'test clf accuracy') 
