
# coding: utf-8

# In[9]:


import os
import random
import tensorflow as tf
import time
import tarfile
import numpy as np
import cv2
import glob

import pdb


# In[10]:


class DeconvNet:
    
    def __init__(self, checkpoint_dir='./checkpoints/'):
        #self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours =1)
        #self.saver = tf.train.Saver()
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        self.session = tf.Session(config = config)
        self.session.run(tf.global_variables_initializer())
        self.checkpoint_dir = checkpoint_dir
        
    def weight_variable(self, shape):
        """
        Create a Weight tensor
        argument:
            shape : The shape of Weight
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(self, shape):
        """
        Create a Bias tensor
        argument:
            shape : The shape of Bias
        """
        print(shape)
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def conv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
        """
        Create a convolutional layer
        argument:
            x: input tensor, shape = [m,h,w,c]
            W_shape : The shape of filter
            b_shape : The shape of bias
            name : layer's name
            padding : padding , default=SAME
            
        detail:
            stride : all using stride = [1,1,1,1]
            activation funtion : all using relu
        """
        W = self.weight_variable(W_shape)
        b = self.bias_variable([b_shape])
        return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1,1,1,1], padding=padding) + b)
        
    def pool_layer(self, x):
        """
        Pooling layer
        argument:
            x: input tensor
        return:
            pooling_output, pooling_argmax
        detail:
            pooling_argmax: Store the origin position of the max value
        """
        return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    def deconv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
        """
        conv2d_transpose(value, filter, output_shape, strides, padding="SAME", data_format="NHWC", name=None)
        Deconvolutional layer
        argument:
            x : input tensor
            W_shape: the filter shape that is same as the filter coming
            b_shape: the bias shape that is same as the bias coming
            name: layer's name
        """
        W = self.weight_variable(W_shape)
        b = self.bias_variable([b_shape])

        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])

        return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b
        
    def unravel_argmax(self, argmax, shape):
        """
        I don't get it
        I get it a little bit LOL 3/5
        """
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.stack(output_list)

    def unpool_layer2x2(self, x, raveled_argmax, out_shape):
        """
        I don't get it
        """
        argmax = self.unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

        t2 = tf.squeeze(argmax)
        t2 = tf.stack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

        t = tf.concat([t2, t1], 3)
        indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)
    
    def build(self):
        """
        Build up the model : DevconvNet 
        placeholder:
            self.x : input image
            self.y : labeled image
            self.rate : learning rate 
        """
        #self.x = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        #self.y = tf.placeholder(tf.int64, shape=(1, None, None))
        #expected = tf.expand_dims(self.y, -1)
        """
        Change self.x self.y 's type to tf.float32
        fixed the height and weight to (512, 256)
        """
        self.x = tf.placeholder(tf.float32, shape=(None, 256, 512, 3))
        self.y = tf.placeholder(tf.float32, shape=(None, 256, 512, 5))
        print("x_shape : " + str(self.x.shape))
        print("y_shape : " + str(self.y.shape))
        expected = self.y
        self.rate = tf.placeholder(tf.float32, shape=[])

        conv_1_1 = self.conv_layer(self.x, [3, 3, 3, 64], 64, 'conv_1_1')
        conv_1_2 = self.conv_layer(conv_1_1, [3, 3, 64, 64], 64, 'conv_1_2')

        pool_1, pool_1_argmax = self.pool_layer(conv_1_2)

        conv_2_1 = self.conv_layer(pool_1, [3, 3, 64, 128], 128, 'conv_2_1')
        conv_2_2 = self.conv_layer(conv_2_1, [3, 3, 128, 128], 128, 'conv_2_2')

        pool_2, pool_2_argmax = self.pool_layer(conv_2_2)

        conv_3_1 = self.conv_layer(pool_2, [3, 3, 128, 256], 256, 'conv_3_1')
        conv_3_2 = self.conv_layer(conv_3_1, [3, 3, 256, 256], 256, 'conv_3_2')
        conv_3_3 = self.conv_layer(conv_3_2, [3, 3, 256, 256], 256, 'conv_3_3')

        pool_3, pool_3_argmax = self.pool_layer(conv_3_3)

        conv_4_1 = self.conv_layer(pool_3, [3, 3, 256, 512], 512, 'conv_4_1')
        conv_4_2 = self.conv_layer(conv_4_1, [3, 3, 512, 512], 512, 'conv_4_2')
        conv_4_3 = self.conv_layer(conv_4_2, [3, 3, 512, 512], 512, 'conv_4_3')

        pool_4, pool_4_argmax = self.pool_layer(conv_4_3)

        conv_5_1 = self.conv_layer(pool_4, [3, 3, 512, 512], 512, 'conv_5_1')
        conv_5_2 = self.conv_layer(conv_5_1, [3, 3, 512, 512], 512, 'conv_5_2')
        conv_5_3 = self.conv_layer(conv_5_2, [3, 3, 512, 512], 512, 'conv_5_3')

        pool_5, pool_5_argmax = self.pool_layer(conv_5_3)

        fc_6 = self.conv_layer(pool_5, [7, 7, 512, 4096], 4096, 'fc_6')
        fc_7 = self.conv_layer(fc_6, [1, 1, 4096, 4096], 4096, 'fc_7')

        deconv_fc_6 = self.deconv_layer(fc_7, [7, 7, 512, 4096], 512, 'fc6_deconv')

        unpool_5 = self.unpool_layer2x2(deconv_fc_6, pool_5_argmax, tf.shape(conv_5_3))

        deconv_5_3 = self.deconv_layer(unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3')
        deconv_5_2 = self.deconv_layer(deconv_5_3, [3, 3, 512, 512], 512, 'deconv_5_2')
        deconv_5_1 = self.deconv_layer(deconv_5_2, [3, 3, 512, 512], 512, 'deconv_5_1')

        unpool_4 = self.unpool_layer2x2(deconv_5_1, pool_4_argmax, tf.shape(conv_4_3))

        deconv_4_3 = self.deconv_layer(unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3')
        deconv_4_2 = self.deconv_layer(deconv_4_3, [3, 3, 512, 512], 512, 'deconv_4_2')
        deconv_4_1 = self.deconv_layer(deconv_4_2, [3, 3, 256, 512], 256, 'deconv_4_1')

        unpool_3 = self.unpool_layer2x2(deconv_4_1, pool_3_argmax, tf.shape(conv_3_3))

        deconv_3_3 = self.deconv_layer(unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3')
        deconv_3_2 = self.deconv_layer(deconv_3_3, [3, 3, 256, 256], 256, 'deconv_3_2')
        deconv_3_1 = self.deconv_layer(deconv_3_2, [3, 3, 128, 256], 128, 'deconv_3_1')

        unpool_2 = self.unpool_layer2x2(deconv_3_1, pool_2_argmax, tf.shape(conv_2_2))

        deconv_2_2 = self.deconv_layer(unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2')
        deconv_2_1 = self.deconv_layer(deconv_2_2, [3, 3, 64, 128], 64, 'deconv_2_1')

        unpool_1 = self.unpool_layer2x2(deconv_2_1, pool_1_argmax, tf.shape(conv_1_2))

        deconv_1_2 = self.deconv_layer(unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2')
        deconv_1_1 = self.deconv_layer(deconv_1_2, [3, 3, 32, 64], 32, 'deconv_1_1')

        score_1 = self.deconv_layer(deconv_1_1, [1, 1, 5, 32], 5, 'score_1')

        logits = tf.reshape(score_1, (-1, 5)) # flatten the score_1
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(expected, [-1]), logits=logits, name='x_entropy')
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(expected, [-1,5]), logits=logits, name='cross_entropy')
        self.loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        self.train_step = tf.train.AdamOptimizer(self.rate).minimize(self.loss)

        self.prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(score_1)), dimension=3)
        #self.accuracy = tf.reduce_sum(tf.pow(self.prediction - expected, 2))
    
    def train(self, train_stage=1, training_steps=5, restore_session=False, learning_rate=1e-6):
        self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours =1)
        if restore_session:
            step_start = restore_session()
        else:
            step_start = 0

        if train_stage == 1:
            """
            feed in train data
            """
            x_path = "./dataset/preprocess_image/x"
            y_path = "./dataset/preprocess_image/ys.npy"
            x_lists = glob.glob(x_path + "/*.png")
            y_lists = glob.glob(y_path + "/*.npy")
            trainset = [(a,b) for a,b in zip(x_lists, y_lists)]
        else:
            """
            feed in dev data
            """
            trainset = None

        for i in range(step_start, step_start+training_steps):
            
            """
            # pick random line from file
            random_line = random.choice(trainset)
            image_file = random_line.split(' ')[0]
            ground_truth_file = random_line.split(' ')[1]
            image = np.float32(cv2.imread('data' + image_file))
            ground_truth = cv2.imread('data' + ground_truth_file[:-1], cv2.IMREAD_GRAYSCALE)
            #
            # norm to 21 classes [0-20] (see paper)
            ground_truth = (ground_truth / 255) * 20
            """
            """
            4/7 I don't know how many pics should I feed
            """
            """
            image = np.float32(cv2.imread("./dataset/preprocess_image/x/bremen_000002_000019_leftImg8bit.png"))

            ground_truth = np.float32(np.load("./dataset/preprocess_image/ys.npy/bremen_000002_000019_gtFine_color.png.npy"))
            """
            image = np.zeros((len(x_lists), 256, 512, 3)) # batch size = 78 for testing the code (batch size, h, w, 3)
            ground_truth = np.zeros((len(y_lists), 256, 512, 5))  # (batch size, h, w , classes)
            random.shuffle(x_lists)
            random.shuffle(y_lists)
            for j in range(len(x_lists)):
                img = cv2.imread(x_lists[j])
                print(img.shape)
                image[j,:,:,:] = img
            assert image.shape == (len(x_lists), 256, 512, 3)
            image = np.float32(image)

    

            for j in range(len(y_lists)):
                ground_truth[j,:,:,:] = np.float32(np.load(y_lists[j]))
            assert ground_truth.shape == (len(y_lists), 256, 512, 5) 
            ground_truth = np.float32(ground_truth)

            print('run train step: '+str(i))
            start = time.time()
            self.train_step.run(session=self.session, feed_dict={self.x: image, self.y: ground_truth, self.rate: learning_rate})
            

            if i % 10000 == 0:
                print('step {} finished in {:.2f} s with loss of {:.6f}'.format(i, time.time() - start, self.loss.eval(session=self.session, feed_dict={self.x: [image], self.y: [ground_truth]})))
                self.saver.save(self.session, self.checkpoint_dir+'model', global_step=i)
                print('Model {} saved'.format(i))

def Generate_test_tensor():
    filename_queue = tf.train.string_input_producer(['/dataset/preprocess_image/x']) #  list of files to read

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a = sess.run(my_img)
        print(a)
    return my_img

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  #image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

def Generate_test_tensor_v2():
    """
    base on tensorflow dataset api
    """
    """
    1 traveling ./dataset/preprocess_image/x/ to get all the files's name
    2 pass the list to tf.constant() as filenames
    """
    dealing_path = './dataset/preprocess_image/x/'
    dealing_list = []
    for (dirpath, dirnames, filenames) in os.walk(dealing_path):
        dealing_list.extend(filenames)
    filenames = tf.constant(dealing_list)
    """
    I have to figure out some way to get all the labels as the following format
    1 traveling ./dataset/preprocess_image/ys.npy/ to get all the files's name
    2 np.load the array
    """
    #labels = tf.constant([0, 37, ...])
    
    dealing_path = './dataset/preprocess_image/ys.npy/'
    dealing_list = []
    for (dirpath, dirnames, filenames) in walk(dealing_path):
        dealing_list.extend(filenames)

    for dealing_npy in dealing_list:
        np.load(dealing_path + dealing_y)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)

if __name__ == '__main__':
    test_model = DeconvNet()

    with tf.device('/cpu:0'):
        test_model.build()
        test_model.train()
    #x_path = "./dataset/preprocess_image/x"
    #y_path = "./dataset/preprocess_image/ys.npy"
    #x_lists = glob.glob(x_path + "/*.png")
    #y_lists = glob.glob(y_path + "/*.npy")
    #print(x_lists)
    #image = cv2.imread(x_lists[1])
    #print(image.shape)
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #Generate_test_tensor()