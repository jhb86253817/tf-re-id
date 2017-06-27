# building five cnn architectures 
from __future__ import division
import tensorflow as tf
import numpy as np

def bias_variable(shape):
    """Initialization of bias term."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    """Definition of convolutional operator."""
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool(x):
    """Definition of max-pooling."""
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def center_loss(features, label, label_stats, centers, alfa):
    """The center loss.
       features: [batch_size, 512], the embedding of images. 
       label: [batch_size, class_num], class label, the label index is 1, others are 0.
       labels_stats: [batch_size, 1], the count of each label in the batch.
       centers: [class_num, 512], center points, each class have one.
       alfa: float, updating rate of centers.
    """
    label = tf.arg_max(label, 1)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = alfa * (centers_batch - features)
    diff = diff / label_stats
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.nn.l2_loss(features - centers_batch)
    return loss, centers

def cnn_i(images_placeholder, labels_placeholder, id_num, alpha):
    """cnn with only identification loss."""
    w_conv1 = tf.get_variable('w_conv1', shape=[3, 3, 3, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=1))
    b_conv1 = bias_variable([32])
    h_conv1 = conv2d(images_placeholder, w_conv1) 
    h_conv1 = tf.contrib.layers.batch_norm(h_conv1) + b_conv1
    # leaky relu
    h_conv1 = tf.maximum(0.2*h_conv1, h_conv1)
    
    w_conv2 = tf.get_variable('w_conv2', shape=[3, 3, 32, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=2))
    b_conv2 = bias_variable([32])
    h_conv2 = conv2d(h_conv1, w_conv2) 
    h_conv2 = tf.contrib.layers.batch_norm(h_conv2) + b_conv2
    h_conv2 = tf.maximum(0.2*h_conv2, h_conv2)
    
    w_conv3 = tf.get_variable('w_conv3', shape=[3, 3, 32, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=3))
    b_conv3 = bias_variable([32])
    h_conv3 = conv2d(h_conv2, w_conv3) 
    h_conv3 = tf.contrib.layers.batch_norm(h_conv3) + b_conv3
    h_conv3 = tf.maximum(0.2*h_conv3, h_conv3)
    h_pool3 = max_pool(h_conv3)
    
    w_conv4 = tf.get_variable('w_conv4', shape=[3, 3, 32, 64],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=4))
    b_conv4 = bias_variable([64])
    h_conv4 = conv2d(h_pool3, w_conv4) 
    h_conv4 = tf.contrib.layers.batch_norm(h_conv4) + b_conv4
    h_conv4 = tf.maximum(0.2*h_conv4, h_conv4)
    
    w_conv5 = tf.get_variable('w_conv5', shape=[3, 3, 64, 96],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=5))
    b_conv5 = bias_variable([96])
    h_conv5 = conv2d(h_conv4, w_conv5) 
    h_conv5 = tf.contrib.layers.batch_norm(h_conv5) + b_conv5
    h_conv5 = tf.maximum(0.2*h_conv5, h_conv5)
    h_pool5 = max_pool(h_conv5)
    
    w_conv6 = tf.get_variable('w_conv6', shape=[3, 3, 96, 128],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=6))
    b_conv6 = bias_variable([128])
    h_conv6 = conv2d(h_pool5, w_conv6) 
    h_conv6 = tf.contrib.layers.batch_norm(h_conv6) + b_conv6
    h_conv6 = tf.maximum(0.2*h_conv6, h_conv6)
    
    w_conv7 = tf.get_variable('w_conv7', shape=[3, 3, 128, 192],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=7))
    b_conv7 = bias_variable([192])
    h_conv7 = conv2d(h_conv6, w_conv7) 
    h_conv7 = tf.contrib.layers.batch_norm(h_conv7) + b_conv7
    h_conv7 = tf.maximum(0.2*h_conv7, h_conv7)
    h_pool7 = max_pool(h_conv7)
    
    w_conv8 = tf.get_variable('w_conv8', shape=[3, 3, 192, 256],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=8))
    b_conv8 = bias_variable([256])
    h_conv8 = conv2d(h_pool7, w_conv8) 
    h_conv8 = tf.contrib.layers.batch_norm(h_conv8) + b_conv8
    h_conv8 = tf.maximum(0.2*h_conv8, h_conv8)
    
    w_conv9 = tf.get_variable('w_conv9', shape=[3, 3, 256, 384],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=9))
    b_conv9 = bias_variable([384])
    h_conv9 = conv2d(h_conv8, w_conv9) 
    h_conv9 = tf.contrib.layers.batch_norm(h_conv9) + b_conv9
    h_conv9 = tf.maximum(0.2*h_conv9, h_conv9)
    h_pool9 = max_pool(h_conv9)
    
    w_fc1 = tf.get_variable('w_fc1', shape=[8*3*384, 512],
                            initializer=tf.contrib.layers.variance_scaling_initializer(seed=10))
    b_fc1 = bias_variable([512])
    h_pool9_flat = tf.reshape(h_pool9, [-1, 8*3*384])
    h_fc1 = tf.matmul(h_pool9_flat, w_fc1)
    h_fc1 = tf.contrib.layers.batch_norm(h_fc1) + b_fc1
    h_fc1 = tf.maximum(0.2*h_fc1, h_fc1)
    
    w_fc2 = tf.get_variable('w_fc2', shape=[512, id_num],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=13))
    logits = tf.matmul(h_fc1, w_fc2)  
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels_placeholder, name='xentropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    # weight decay with l2 norm
    w_conv1_l2 = alpha * tf.nn.l2_loss(w_conv1)
    w_conv2_l2 = alpha * tf.nn.l2_loss(w_conv2)
    w_conv3_l2 = alpha * tf.nn.l2_loss(w_conv3)
    w_conv4_l2 = alpha * tf.nn.l2_loss(w_conv4)
    w_conv5_l2 = alpha * tf.nn.l2_loss(w_conv5)
    w_conv6_l2 = alpha * tf.nn.l2_loss(w_conv6)
    w_conv7_l2 = alpha * tf.nn.l2_loss(w_conv7)
    w_conv8_l2 = alpha * tf.nn.l2_loss(w_conv8)
    w_conv9_l2 = alpha * tf.nn.l2_loss(w_conv9)
    w_fc1_l2 = alpha * tf.nn.l2_loss(w_fc1)
    w_fc2_l2 = alpha * tf.nn.l2_loss(w_fc2)

    loss = cross_entropy_mean + w_conv1_l2 + w_conv2_l2 + w_conv3_l2 + w_conv4_l2 + w_conv5_l2 + w_conv6_l2 + w_conv7_l2 + w_conv8_l2 + w_conv9_l2 + w_fc1_l2 + w_fc2_l2 
    return loss, h_fc1, w_fc2 

def cnn_ic(images_placeholder, labels_placeholder, labels_placeholder_stats, id_num, alpha, landa, alfa, batch_size):
    """cnn with identification loss and center loss."""
    centers = tf.get_variable('centers', [id_num, 512], dtype=tf.float32,
              initializer=tf.constant_initializer(0), trainable=False)

    w_conv1 = tf.get_variable('w_conv1', shape=[3, 3, 3, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=1))
    b_conv1 = bias_variable([32])
    h_conv1 = conv2d(images_placeholder, w_conv1) 
    h_conv1 = tf.contrib.layers.batch_norm(h_conv1) + b_conv1
    # leaky relu
    h_conv1 = tf.maximum(0.2*h_conv1, h_conv1)
    
    w_conv2 = tf.get_variable('w_conv2', shape=[3, 3, 32, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=2))
    b_conv2 = bias_variable([32])
    h_conv2 = conv2d(h_conv1, w_conv2) 
    h_conv2 = tf.contrib.layers.batch_norm(h_conv2) + b_conv2
    h_conv2 = tf.maximum(0.2*h_conv2, h_conv2)
    
    w_conv3 = tf.get_variable('w_conv3', shape=[3, 3, 32, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=3))
    b_conv3 = bias_variable([32])
    h_conv3 = conv2d(h_conv2, w_conv3) 
    h_conv3 = tf.contrib.layers.batch_norm(h_conv3) + b_conv3
    h_conv3 = tf.maximum(0.2*h_conv3, h_conv3)
    h_pool3 = max_pool(h_conv3)
    
    w_conv4 = tf.get_variable('w_conv4', shape=[3, 3, 32, 64],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=4))
    b_conv4 = bias_variable([64])
    h_conv4 = conv2d(h_pool3, w_conv4) 
    h_conv4 = tf.contrib.layers.batch_norm(h_conv4) + b_conv4
    h_conv4 = tf.maximum(0.2*h_conv4, h_conv4)
    
    w_conv5 = tf.get_variable('w_conv5', shape=[3, 3, 64, 96],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=5))
    b_conv5 = bias_variable([96])
    h_conv5 = conv2d(h_conv4, w_conv5) 
    h_conv5 = tf.contrib.layers.batch_norm(h_conv5) + b_conv5
    h_conv5 = tf.maximum(0.2*h_conv5, h_conv5)
    h_pool5 = max_pool(h_conv5)
    
    w_conv6 = tf.get_variable('w_conv6', shape=[3, 3, 96, 128],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=6))
    b_conv6 = bias_variable([128])
    h_conv6 = conv2d(h_pool5, w_conv6) 
    h_conv6 = tf.contrib.layers.batch_norm(h_conv6) + b_conv6
    h_conv6 = tf.maximum(0.2*h_conv6, h_conv6)
    
    w_conv7 = tf.get_variable('w_conv7', shape=[3, 3, 128, 192],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=7))
    b_conv7 = bias_variable([192])
    h_conv7 = conv2d(h_conv6, w_conv7) 
    h_conv7 = tf.contrib.layers.batch_norm(h_conv7) + b_conv7
    h_conv7 = tf.maximum(0.2*h_conv7, h_conv7)
    h_pool7 = max_pool(h_conv7)
    
    w_conv8 = tf.get_variable('w_conv8', shape=[3, 3, 192, 256],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=8))
    b_conv8 = bias_variable([256])
    h_conv8 = conv2d(h_pool7, w_conv8) 
    h_conv8 = tf.contrib.layers.batch_norm(h_conv8) + b_conv8
    h_conv8 = tf.maximum(0.2*h_conv8, h_conv8)
    
    w_conv9 = tf.get_variable('w_conv9', shape=[3, 3, 256, 384],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=9))
    b_conv9 = bias_variable([384])
    h_conv9 = conv2d(h_conv8, w_conv9) 
    h_conv9 = tf.contrib.layers.batch_norm(h_conv9) + b_conv9
    h_conv9 = tf.maximum(0.2*h_conv9, h_conv9)
    h_pool9 = max_pool(h_conv9)
    
    w_fc1 = tf.get_variable('w_fc1', shape=[8*3*384, 512],
                            initializer=tf.contrib.layers.variance_scaling_initializer(seed=10))
    b_fc1 = bias_variable([512])
    h_pool9_flat = tf.reshape(h_pool9, [-1, 8*3*384])
    h_fc1 = tf.matmul(h_pool9_flat, w_fc1)
    h_fc1 = tf.contrib.layers.batch_norm(h_fc1) + b_fc1
    h_fc1 = tf.maximum(0.2*h_fc1, h_fc1)
    
    w_fc2 = tf.get_variable('w_fc2', shape=[512, id_num],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=13))
    logits = tf.matmul(h_fc1, w_fc2)  
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels_placeholder, name='xentropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    # compute center loss, and update centers
    cl, centers_updated = center_loss(h_fc1, labels_placeholder, labels_placeholder_stats, centers, alfa)
    cl_mean = cl / batch_size
    centers = centers_updated

    # weight decay with l2 norm
    w_conv1_l2 = alpha * tf.nn.l2_loss(w_conv1)
    w_conv2_l2 = alpha * tf.nn.l2_loss(w_conv2)
    w_conv3_l2 = alpha * tf.nn.l2_loss(w_conv3)
    w_conv4_l2 = alpha * tf.nn.l2_loss(w_conv4)
    w_conv5_l2 = alpha * tf.nn.l2_loss(w_conv5)
    w_conv6_l2 = alpha * tf.nn.l2_loss(w_conv6)
    w_conv7_l2 = alpha * tf.nn.l2_loss(w_conv7)
    w_conv8_l2 = alpha * tf.nn.l2_loss(w_conv8)
    w_conv9_l2 = alpha * tf.nn.l2_loss(w_conv9)
    w_fc1_l2 = alpha * tf.nn.l2_loss(w_fc1)
    w_fc2_l2 = alpha * tf.nn.l2_loss(w_fc2)
    
    loss = cross_entropy_mean + landa*cl_mean + w_conv1_l2 + w_conv2_l2 + w_conv3_l2 + w_conv4_l2 + w_conv5_l2 + w_conv6_l2 + w_conv7_l2 + w_conv8_l2 + w_conv9_l2 + w_fc1_l2 + w_fc2_l2 
    return loss, h_fc1, centers, w_fc2 

def cnn_frw_ic(images_placeholder, labels_placeholder, labels_placeholder_stats, id_num, alpha, landa, alfa, beta, C, batch_size):
    """cnn with identification loss and center loss, 
       also contains a feature reweighting(frw) layer."""
    centers = tf.get_variable('centers', [id_num, 512], dtype=tf.float32,
              initializer=tf.constant_initializer(0), trainable=False)

    w_conv1 = tf.get_variable('w_conv1', shape=[3, 3, 3, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=1))
    b_conv1 = bias_variable([32])
    h_conv1 = conv2d(images_placeholder, w_conv1) 
    h_conv1 = tf.contrib.layers.batch_norm(h_conv1) + b_conv1
    # leaky relu
    h_conv1 = tf.maximum(0.2*h_conv1, h_conv1)
    
    w_conv2 = tf.get_variable('w_conv2', shape=[3, 3, 32, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=2))
    b_conv2 = bias_variable([32])
    h_conv2 = conv2d(h_conv1, w_conv2) 
    h_conv2 = tf.contrib.layers.batch_norm(h_conv2) + b_conv2
    h_conv2 = tf.maximum(0.2*h_conv2, h_conv2)
    
    w_conv3 = tf.get_variable('w_conv3', shape=[3, 3, 32, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=3))
    b_conv3 = bias_variable([32])
    h_conv3 = conv2d(h_conv2, w_conv3) 
    h_conv3 = tf.contrib.layers.batch_norm(h_conv3) + b_conv3
    h_conv3 = tf.maximum(0.2*h_conv3, h_conv3)
    h_pool3 = max_pool(h_conv3)
    
    w_conv4 = tf.get_variable('w_conv4', shape=[3, 3, 32, 64],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=4))
    b_conv4 = bias_variable([64])
    h_conv4 = conv2d(h_pool3, w_conv4) 
    h_conv4 = tf.contrib.layers.batch_norm(h_conv4) + b_conv4
    h_conv4 = tf.maximum(0.2*h_conv4, h_conv4)
    
    w_conv5 = tf.get_variable('w_conv5', shape=[3, 3, 64, 96],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=5))
    b_conv5 = bias_variable([96])
    h_conv5 = conv2d(h_conv4, w_conv5) 
    h_conv5 = tf.contrib.layers.batch_norm(h_conv5) + b_conv5
    h_conv5 = tf.maximum(0.2*h_conv5, h_conv5)
    h_pool5 = max_pool(h_conv5)
    
    w_conv6 = tf.get_variable('w_conv6', shape=[3, 3, 96, 128],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=6))
    b_conv6 = bias_variable([128])
    h_conv6 = conv2d(h_pool5, w_conv6) 
    h_conv6 = tf.contrib.layers.batch_norm(h_conv6) + b_conv6
    h_conv6 = tf.maximum(0.2*h_conv6, h_conv6)
    
    w_conv7 = tf.get_variable('w_conv7', shape=[3, 3, 128, 192],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=7))
    b_conv7 = bias_variable([192])
    h_conv7 = conv2d(h_conv6, w_conv7) 
    h_conv7 = tf.contrib.layers.batch_norm(h_conv7) + b_conv7
    h_conv7 = tf.maximum(0.2*h_conv7, h_conv7)
    h_pool7 = max_pool(h_conv7)
    
    w_conv8 = tf.get_variable('w_conv8', shape=[3, 3, 192, 256],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=8))
    b_conv8 = bias_variable([256])
    h_conv8 = conv2d(h_pool7, w_conv8) 
    h_conv8 = tf.contrib.layers.batch_norm(h_conv8) + b_conv8
    h_conv8 = tf.maximum(0.2*h_conv8, h_conv8)
    
    w_conv9 = tf.get_variable('w_conv9', shape=[3, 3, 256, 384],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=9))
    b_conv9 = bias_variable([384])
    h_conv9 = conv2d(h_conv8, w_conv9) 
    h_conv9 = tf.contrib.layers.batch_norm(h_conv9) + b_conv9
    h_conv9 = tf.maximum(0.2*h_conv9, h_conv9)
    h_pool9 = max_pool(h_conv9)
    
    w_fc1 = tf.get_variable('w_fc1', shape=[8*3*384, 512],
                            initializer=tf.contrib.layers.variance_scaling_initializer(seed=10))
    b_fc1 = bias_variable([512])
    h_pool9_flat = tf.reshape(h_pool9, [-1, 8*3*384])
    h_fc1 = tf.matmul(h_pool9_flat, w_fc1)
    h_fc1 = tf.contrib.layers.batch_norm(h_fc1) + b_fc1
    h_fc1 = tf.maximum(0.2*h_fc1, h_fc1)

    # frw layer
    w_coef = tf.get_variable('w_coef', shape=[512],
                            initializer=tf.contrib.layers.variance_scaling_initializer(seed=11))
    h_fc1 = h_fc1 * w_coef
    
    w_fc2 = tf.get_variable('w_fc2', shape=[512, id_num],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=13))
    logits = tf.matmul(h_fc1, w_fc2)  
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels_placeholder, name='xentropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    # compute center loss, and update centers
    cl, centers_updated = center_loss(h_fc1, labels_placeholder, labels_placeholder_stats, centers, alfa)
    cl_mean = cl / batch_size
    centers = centers_updated

    # weight decay with l2 norm
    w_conv1_l2 = alpha * tf.nn.l2_loss(w_conv1)
    w_conv2_l2 = alpha * tf.nn.l2_loss(w_conv2)
    w_conv3_l2 = alpha * tf.nn.l2_loss(w_conv3)
    w_conv4_l2 = alpha * tf.nn.l2_loss(w_conv4)
    w_conv5_l2 = alpha * tf.nn.l2_loss(w_conv5)
    w_conv6_l2 = alpha * tf.nn.l2_loss(w_conv6)
    w_conv7_l2 = alpha * tf.nn.l2_loss(w_conv7)
    w_conv8_l2 = alpha * tf.nn.l2_loss(w_conv8)
    w_conv9_l2 = alpha * tf.nn.l2_loss(w_conv9)
    w_fc1_l2 = alpha * tf.nn.l2_loss(w_fc1)
    w_fc2_l2 = alpha * tf.nn.l2_loss(w_fc2)
    w_coef_l2 = beta * tf.square(tf.nn.l2_loss(w_coef)-C)
    
    loss = cross_entropy_mean + landa*cl_mean + w_conv1_l2 + w_conv2_l2 + w_conv3_l2 + w_conv4_l2 + w_conv5_l2 + w_conv6_l2 + w_conv7_l2 + w_conv8_l2 + w_conv9_l2 + w_fc1_l2 + w_fc2_l2 + w_coef_l2
    return loss, h_fc1, centers, w_fc2

def cnn_fc_ic(images_placeholder, labels_placeholder, labels_placeholder_stats, id_num, alpha, landa, alfa, batch_size):
    """cnn with identification loss and center loss, 
       also contains one more fully connected layer."""
    centers = tf.get_variable('centers', [id_num, 512], dtype=tf.float32,
              initializer=tf.constant_initializer(0), trainable=False)

    w_conv1 = tf.get_variable('w_conv1', shape=[3, 3, 3, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=1))
    b_conv1 = bias_variable([32])
    h_conv1 = conv2d(images_placeholder, w_conv1) 
    h_conv1 = tf.contrib.layers.batch_norm(h_conv1) + b_conv1
    # leaky relu
    h_conv1 = tf.maximum(0.2*h_conv1, h_conv1)
    
    w_conv2 = tf.get_variable('w_conv2', shape=[3, 3, 32, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=2))
    b_conv2 = bias_variable([32])
    h_conv2 = conv2d(h_conv1, w_conv2) 
    h_conv2 = tf.contrib.layers.batch_norm(h_conv2) + b_conv2
    h_conv2 = tf.maximum(0.2*h_conv2, h_conv2)
    
    w_conv3 = tf.get_variable('w_conv3', shape=[3, 3, 32, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=3))
    b_conv3 = bias_variable([32])
    h_conv3 = conv2d(h_conv2, w_conv3) 
    h_conv3 = tf.contrib.layers.batch_norm(h_conv3) + b_conv3
    h_conv3 = tf.maximum(0.2*h_conv3, h_conv3)
    h_pool3 = max_pool(h_conv3)
    
    w_conv4 = tf.get_variable('w_conv4', shape=[3, 3, 32, 64],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=4))
    b_conv4 = bias_variable([64])
    h_conv4 = conv2d(h_pool3, w_conv4) 
    h_conv4 = tf.contrib.layers.batch_norm(h_conv4) + b_conv4
    h_conv4 = tf.maximum(0.2*h_conv4, h_conv4)
    
    w_conv5 = tf.get_variable('w_conv5', shape=[3, 3, 64, 96],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=5))
    b_conv5 = bias_variable([96])
    h_conv5 = conv2d(h_conv4, w_conv5) 
    h_conv5 = tf.contrib.layers.batch_norm(h_conv5) + b_conv5
    h_conv5 = tf.maximum(0.2*h_conv5, h_conv5)
    h_pool5 = max_pool(h_conv5)
    
    w_conv6 = tf.get_variable('w_conv6', shape=[3, 3, 96, 128],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=6))
    b_conv6 = bias_variable([128])
    h_conv6 = conv2d(h_pool5, w_conv6) 
    h_conv6 = tf.contrib.layers.batch_norm(h_conv6) + b_conv6
    h_conv6 = tf.maximum(0.2*h_conv6, h_conv6)
    
    w_conv7 = tf.get_variable('w_conv7', shape=[3, 3, 128, 192],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=7))
    b_conv7 = bias_variable([192])
    h_conv7 = conv2d(h_conv6, w_conv7) 
    h_conv7 = tf.contrib.layers.batch_norm(h_conv7) + b_conv7
    h_conv7 = tf.maximum(0.2*h_conv7, h_conv7)
    h_pool7 = max_pool(h_conv7)
    
    w_conv8 = tf.get_variable('w_conv8', shape=[3, 3, 192, 256],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=8))
    b_conv8 = bias_variable([256])
    h_conv8 = conv2d(h_pool7, w_conv8) 
    h_conv8 = tf.contrib.layers.batch_norm(h_conv8) + b_conv8
    h_conv8 = tf.maximum(0.2*h_conv8, h_conv8)
    
    w_conv9 = tf.get_variable('w_conv9', shape=[3, 3, 256, 384],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=9))
    b_conv9 = bias_variable([384])
    h_conv9 = conv2d(h_conv8, w_conv9) 
    h_conv9 = tf.contrib.layers.batch_norm(h_conv9) + b_conv9
    h_conv9 = tf.maximum(0.2*h_conv9, h_conv9)
    h_pool9 = max_pool(h_conv9)
    
    w_fc1 = tf.get_variable('w_fc1', shape=[8*3*384, 512],
                            initializer=tf.contrib.layers.variance_scaling_initializer(seed=10))
    b_fc1 = bias_variable([512])
    h_pool9_flat = tf.reshape(h_pool9, [-1, 8*3*384])
    h_fc1 = tf.matmul(h_pool9_flat, w_fc1)
    h_fc1 = tf.contrib.layers.batch_norm(h_fc1) + b_fc1
    h_fc1 = tf.maximum(0.2*h_fc1, h_fc1)
    
    w_fc2 = tf.get_variable('w_fc2', shape=[512, 512],
                            initializer=tf.contrib.layers.variance_scaling_initializer(seed=11))
    b_fc2 = bias_variable([512])
    h_fc2 = tf.matmul(h_fc1, w_fc2) 
    h_fc2 = tf.contrib.layers.batch_norm(h_fc2) + b_fc2
    h_fc2 = tf.maximum(0.2*h_fc2, h_fc2)

    w_fc3 = tf.get_variable('w_fc3', shape=[512, id_num],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=13))
    logits = tf.matmul(h_fc2, w_fc3)  
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels_placeholder, name='xentropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    # computer center loss, and update centers
    cl, centers_updated = center_loss(h_fc2, labels_placeholder, labels_placeholder_stats, centers, alfa)
    cl_mean = cl / batch_size
    centers = centers_updated

    # weight decay with l2 norm
    w_conv1_l2 = alpha * tf.nn.l2_loss(w_conv1)
    w_conv2_l2 = alpha * tf.nn.l2_loss(w_conv2)
    w_conv3_l2 = alpha * tf.nn.l2_loss(w_conv3)
    w_conv4_l2 = alpha * tf.nn.l2_loss(w_conv4)
    w_conv5_l2 = alpha * tf.nn.l2_loss(w_conv5)
    w_conv6_l2 = alpha * tf.nn.l2_loss(w_conv6)
    w_conv7_l2 = alpha * tf.nn.l2_loss(w_conv7)
    w_conv8_l2 = alpha * tf.nn.l2_loss(w_conv8)
    w_conv9_l2 = alpha * tf.nn.l2_loss(w_conv9)
    w_fc1_l2 = alpha * tf.nn.l2_loss(w_fc1)
    w_fc2_l2 = alpha * tf.nn.l2_loss(w_fc2)
    w_fc3_l2 = alpha * tf.nn.l2_loss(w_fc3)
    
    loss = cross_entropy_mean + landa*cl_mean + w_conv1_l2 + w_conv2_l2 + w_conv3_l2 + w_conv4_l2 + w_conv5_l2 + w_conv6_l2 + w_conv7_l2 + w_conv8_l2 + w_conv9_l2 + w_fc1_l2 + w_fc2_l2 + w_fc3_l2
    return loss, h_fc2, centers, w_fc3

def cnn_iv(images_placeholder_a, images_placeholder_b, labels_placeholder_vs, labels_placeholder_cs_a, labels_placeholder_cs_b, id_num, alpha):
    """cnn with identification loss and verification loss(binary classifier)."""
    w_conv1 = tf.get_variable('w_conv1', shape=[3, 3, 3, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=1))
    b_conv1 = bias_variable([32])
    h_conv1_a = conv2d(images_placeholder_a, w_conv1) 
    h_conv1_a = tf.contrib.layers.batch_norm(h_conv1_a) + b_conv1
    h_conv1_a = tf.maximum(0.2*h_conv1_a, h_conv1_a)
    
    h_conv1_b = conv2d(images_placeholder_b, w_conv1) 
    h_conv1_b = tf.contrib.layers.batch_norm(h_conv1_b) + b_conv1
    h_conv1_b = tf.maximum(0.2*h_conv1_b, h_conv1_b)
    
    w_conv2 = tf.get_variable('w_conv2', shape=[3, 3, 32, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=2))
    b_conv2 = bias_variable([32])
    h_conv2_a = conv2d(h_conv1_a, w_conv2) 
    h_conv2_a = tf.contrib.layers.batch_norm(h_conv2_a) + b_conv2
    h_conv2_a = tf.maximum(0.2*h_conv2_a, h_conv2_a)
    
    h_conv2_b = conv2d(h_conv1_b, w_conv2) 
    h_conv2_b = tf.contrib.layers.batch_norm(h_conv2_b) + b_conv2
    h_conv2_b = tf.maximum(0.2*h_conv2_b, h_conv2_b)
    
    w_conv3 = tf.get_variable('w_conv3', shape=[3, 3, 32, 32],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=3))
    b_conv3 = bias_variable([32])
    h_conv3_a = conv2d(h_conv2_a, w_conv3) 
    h_conv3_a = tf.contrib.layers.batch_norm(h_conv3_a) + b_conv3
    h_conv3_a = tf.maximum(0.2*h_conv3_a, h_conv3_a)
    h_pool3_a = max_pool(h_conv3_a)
    
    h_conv3_b = conv2d(h_conv2_b, w_conv3) 
    h_conv3_b = tf.contrib.layers.batch_norm(h_conv3_b) + b_conv3
    h_conv3_b = tf.maximum(0.2*h_conv3_b, h_conv3_b)
    h_pool3_b = max_pool(h_conv3_b)
    
    w_conv4 = tf.get_variable('w_conv4', shape=[3, 3, 32, 64],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=4))
    b_conv4 = bias_variable([64])
    h_conv4_a = conv2d(h_pool3_a, w_conv4) 
    h_conv4_a = tf.contrib.layers.batch_norm(h_conv4_a) + b_conv4
    h_conv4_a = tf.maximum(0.2*h_conv4_a, h_conv4_a)
    
    h_conv4_b = conv2d(h_pool3_b, w_conv4) 
    h_conv4_b = tf.contrib.layers.batch_norm(h_conv4_b) + b_conv4
    h_conv4_b = tf.maximum(0.2*h_conv4_b, h_conv4_b)

    w_conv5 = tf.get_variable('w_conv5', shape=[3, 3, 64, 96],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=5))
    b_conv5 = bias_variable([96])
    h_conv5_a = conv2d(h_conv4_a, w_conv5) 
    h_conv5_a = tf.contrib.layers.batch_norm(h_conv5_a) + b_conv5
    h_conv5_a = tf.maximum(0.2*h_conv5_a, h_conv5_a)
    h_pool5_a = max_pool(h_conv5_a)
    
    h_conv5_b = conv2d(h_conv4_b, w_conv5) 
    h_conv5_b = tf.contrib.layers.batch_norm(h_conv5_b) + b_conv5
    h_conv5_b = tf.maximum(0.2*h_conv5_b, h_conv5_b)
    h_pool5_b = max_pool(h_conv5_b)

    w_conv6 = tf.get_variable('w_conv6', shape=[3, 3, 96, 128],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=6))
    b_conv6 = bias_variable([128])
    h_conv6_a = conv2d(h_pool5_a, w_conv6) 
    h_conv6_a = tf.contrib.layers.batch_norm(h_conv6_a) + b_conv6
    h_conv6_a = tf.maximum(0.2*h_conv6_a, h_conv6_a)
    
    h_conv6_b = conv2d(h_pool5_b, w_conv6) 
    h_conv6_b = tf.contrib.layers.batch_norm(h_conv6_b) + b_conv6
    h_conv6_b = tf.maximum(0.2*h_conv6_b, h_conv6_b)

    w_conv7 = tf.get_variable('w_conv7', shape=[3, 3, 128, 192],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=7))
    b_conv7 = bias_variable([192])
    h_conv7_a = conv2d(h_conv6_a, w_conv7) 
    h_conv7_a = tf.contrib.layers.batch_norm(h_conv7_a) + b_conv7
    h_conv7_a = tf.maximum(0.2*h_conv7_a, h_conv7_a)
    h_pool7_a = max_pool(h_conv7_a)
    
    h_conv7_b = conv2d(h_conv6_b, w_conv7) 
    h_conv7_b = tf.contrib.layers.batch_norm(h_conv7_b) + b_conv7
    h_conv7_b = tf.maximum(0.2*h_conv7_b, h_conv7_b)
    h_pool7_b = max_pool(h_conv7_b)

    w_conv8 = tf.get_variable('w_conv8', shape=[3, 3, 192, 256],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=8))
    b_conv8 = bias_variable([256])
    h_conv8_a = conv2d(h_pool7_a, w_conv8) 
    h_conv8_a = tf.contrib.layers.batch_norm(h_conv8_a) + b_conv8
    h_conv8_a = tf.maximum(0.2*h_conv8_a, h_conv8_a)
    
    h_conv8_b = conv2d(h_pool7_b, w_conv8) 
    h_conv8_b = tf.contrib.layers.batch_norm(h_conv8_b) + b_conv8
    h_conv8_b = tf.maximum(0.2*h_conv8_b, h_conv8_b)

    w_conv9 = tf.get_variable('w_conv9', shape=[3, 3, 256, 384],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=9))
    b_conv9 = bias_variable([384])
    h_conv9_a = conv2d(h_conv8_a, w_conv9) 
    h_conv9_a = tf.contrib.layers.batch_norm(h_conv9_a) + b_conv9
    h_conv9_a = tf.maximum(0.2*h_conv9_a, h_conv9_a)
    h_pool9_a = max_pool(h_conv9_a)
    
    h_conv9_b = conv2d(h_conv8_b, w_conv9) 
    h_conv9_b = tf.contrib.layers.batch_norm(h_conv9_b) + b_conv9
    h_conv9_b = tf.maximum(0.2*h_conv9_b, h_conv9_b)
    h_pool9_b = max_pool(h_conv9_b)

    w_fc1 = tf.get_variable('w_fc1', shape=[8*3*384, 512],
                            initializer=tf.contrib.layers.variance_scaling_initializer(seed=10))
    b_fc1 = bias_variable([512])
    h_pool9_a_flat = tf.reshape(h_pool9_a, [-1, 8*3*384])
    h_fc1_a = tf.matmul(h_pool9_a_flat, w_fc1)
    h_fc1_a = tf.contrib.layers.batch_norm(h_fc1_a) + b_fc1
    h_fc1_a = tf.maximum(0.2*h_fc1_a, h_fc1_a)

    h_pool9_b_flat = tf.reshape(h_pool9_b, [-1, 8*3*384])
    h_fc1_b = tf.matmul(h_pool9_b_flat, w_fc1)
    h_fc1_b = tf.contrib.layers.batch_norm(h_fc1_b) + b_fc1
    h_fc1_b = tf.maximum(0.2*h_fc1_b, h_fc1_b)
    #####################################
    # verification subnet
    diff = tf.nn.relu(h_fc1_a-h_fc1_b)

    w_fc1_vs = tf.get_variable('w_fc1_vs', shape=[512, 512],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=11))
    b_fc1_vs = bias_variable([512])
    h_fc1_vs = tf.matmul(diff,w_fc1_vs) + b_fc1_vs
    h_fc1_vs = tf.maximum(0.2*h_fc1_vs, h_fc1_vs)

    w_fc2_vs = tf.get_variable('w_fc2_vs', shape=[512, 2],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=12))
    b_fc2_vs = bias_variable([2])
    logits_vs = tf.matmul(h_fc1_vs, w_fc2_vs) + b_fc2_vs
    cross_entropy_vs = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits_vs, labels=labels_placeholder_vs, name='xentropy_vs')
    cross_entropy_vs_mean = tf.reduce_mean(cross_entropy_vs, name='xentropy_mean_vs')
    #####################################
    # classification subnet
    w_fc1_cs = tf.get_variable('w_fc1_cs', shape=[512, id_num],
                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=13))
    logits_cs_a = tf.matmul(h_fc1_a, w_fc1_cs)  
    cross_entropy_cs_a = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits_cs_a, labels=labels_placeholder_cs_a, name='xentropy_cs_a')
    cross_entropy_cs_mean_a = tf.reduce_mean(cross_entropy_cs_a, name='xentropy_mean_cs_a')

    logits_cs_b = tf.matmul(h_fc1_b, w_fc1_cs) 
    cross_entropy_cs_b = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits_cs_b, labels=labels_placeholder_cs_b, name='xentropy_cs_b')
    cross_entropy_cs_mean_b = tf.reduce_mean(cross_entropy_cs_b, name='xentropy_mean_cs_b')
    
    w_conv1_l2 = alpha * tf.nn.l2_loss(w_conv1)
    w_conv2_l2 = alpha * tf.nn.l2_loss(w_conv2)
    w_conv3_l2 = alpha * tf.nn.l2_loss(w_conv3)
    w_conv4_l2 = alpha * tf.nn.l2_loss(w_conv4)
    w_conv5_l2 = alpha * tf.nn.l2_loss(w_conv5)
    w_conv6_l2 = alpha * tf.nn.l2_loss(w_conv6)
    w_conv7_l2 = alpha * tf.nn.l2_loss(w_conv7)
    w_conv8_l2 = alpha * tf.nn.l2_loss(w_conv8)
    w_conv9_l2 = alpha * tf.nn.l2_loss(w_conv9)
    w_fc1_l2 = alpha * tf.nn.l2_loss(w_fc1)
    w_fc1_vs_l2 = alpha * tf.nn.l2_loss(w_fc1_vs)
    w_fc2_vs_l2 = alpha * tf.nn.l2_loss(w_fc2_vs)
    w_fc1_cs_l2 = alpha * tf.nn.l2_loss(w_fc1_cs)

    # loss for pretraining
    loss = 0.5*cross_entropy_cs_mean_a + 0.5*cross_entropy_cs_mean_b + 0.5*cross_entropy_vs_mean + w_conv1_l2 + w_conv2_l2 + w_conv3_l2 + w_conv4_l2 + w_conv5_l2 + w_conv6_l2 + w_conv7_l2 + w_conv8_l2 + w_conv9_l2 + w_fc1_l2 + w_fc1_vs_l2 + w_fc2_vs_l2 + w_fc1_cs_l2
    # this loss will be used in fine tune stage
    loss_cs = 0.5*cross_entropy_cs_mean_a + 0.5*cross_entropy_cs_mean_b + w_conv1_l2 + w_conv2_l2 + w_conv3_l2 + w_conv4_l2 + w_conv5_l2 + w_conv6_l2 + w_conv7_l2 + w_conv8_l2 + w_conv9_l2 + w_fc1_l2 + w_fc1_vs_l2 + w_fc2_vs_l2 + w_fc1_cs_l2
    return loss, h_fc1_a, h_fc1_b, loss_cs, w_fc1_cs
