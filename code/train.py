# training functions for different data and cnn architectures
# also contains testing functions
from __future__ import division
import tensorflow as tf
import numpy as np
from collections import Counter
import process_cuhk03
import process_cuhk01
import process_viper
from cnn import cnn_i, cnn_iv, cnn_ic, cnn_fc_ic, cnn_frw_ic
np.random.seed(1)

def next_batch_single(train_imgs_a, train_imgs_b, batch_size):
    """Sampling a batch for training. This one is for single branch 
       network, i.e. cnn-i, cnn-ic, cnn-frw-ic, cnn-fc-ic."""
    data_length = len(train_imgs_a)
    images_feed_a = []
    index_a = np.random.choice(range(data_length), size=int(batch_size/2))
    for i in index_a:
        j = np.random.randint(len(train_imgs_a[i]))
        images_feed_a.append(train_imgs_a[i][j])
    images_feed_b = []
    index_b = np.random.choice(range(data_length), size=int(batch_size/2))
    for i in index_b:
        j = np.random.randint(len(train_imgs_b[i]))
        images_feed_b.append(train_imgs_b[i][j])
    # labels
    labels_feed_a = np.zeros((int(batch_size/2), data_length))
    for i in range(labels_feed_a.shape[0]):
        labels_feed_a[i,index_a[i]] = 1
    labels_feed_b = np.zeros((int(batch_size/2), data_length))
    for i in range(labels_feed_b.shape[0]):
        labels_feed_b[i,index_b[i]] = 1
    images_feed_a = np.array(images_feed_a)
    images_feed_b = np.array(images_feed_b)
    images_feed = np.concatenate([images_feed_a, images_feed_b], axis=0)
    labels_feed = np.concatenate([labels_feed_a, labels_feed_b], axis=0)
    return images_feed, labels_feed

def next_batch_pair(train_imgs_a, train_imgs_b, batch_size):
    """Sampling a batch for training. This one is for two-branch 
       network, i.e. cnn-iv."""
    imgs_pos_a, imgs_pos_b, labels_vs_pos, labels_cs_a_pos, labels_cs_b_pos = positive_batch(train_imgs_a, train_imgs_b, batch_size)
    imgs_neg_a, imgs_neg_b, labels_vs_neg, labels_cs_a_neg, labels_cs_b_neg = negative_batch(train_imgs_a, train_imgs_b, batch_size)
    imgs_a = np.concatenate([imgs_pos_a, imgs_neg_a], axis=0)
    imgs_b = np.concatenate([imgs_pos_b, imgs_neg_b], axis=0)
    labels_vs = np.concatenate([labels_vs_pos, labels_vs_neg], axis=0)
    labels_cs_a = np.concatenate([labels_cs_a_pos, labels_cs_a_neg], axis=0)
    labels_cs_b = np.concatenate([labels_cs_b_pos, labels_cs_b_neg], axis=0)
    return imgs_a, imgs_b, labels_vs, labels_cs_a, labels_cs_b

def positive_batch(train_imgs_a, train_imgs_b, batch_size):
    """Sampling positive samples for two-branch network."""
    train_feed_a = []
    train_feed_b = []
    data_length = len(train_imgs_a)
    index = np.random.choice(range(data_length), size=int(batch_size/2))
    for i in index:
        i_a = np.random.randint(len(train_imgs_a[i]))
        i_b = np.random.randint(len(train_imgs_b[i]))
        train_feed_a.append(train_imgs_a[i][i_a])
        train_feed_b.append(train_imgs_b[i][i_b])
    #########################################################    
    # labels for verification loss
    train_labels_vs_pos = np.zeros((int(batch_size/2), 2))
    # positive label: 10
    train_labels_vs_pos[:,0] = 1
    #########################################################    
    # labels for classification loss
    train_labels_cs_a_pos = np.zeros((int(batch_size/2), data_length))
    train_labels_cs_b_pos = np.zeros((int(batch_size/2), data_length))
    for i in range(train_labels_cs_a_pos.shape[0]):
        train_labels_cs_a_pos[i,index[i]] = 1
        train_labels_cs_b_pos[i,index[i]] = 1
    #########################################################    
    return np.array(train_feed_a), np.array(train_feed_b), train_labels_vs_pos, train_labels_cs_a_pos, train_labels_cs_b_pos

def negative_batch(train_imgs_a, train_imgs_b, batch_size):
    """Sampling negative samples for two-branch network."""
    train_feed_a = []
    train_feed_b = []
    data_length = len(train_imgs_a)
    index_a = []
    index_b = []
    for i in range(int(batch_size/2)):
        index = np.random.choice(range(data_length), size=2, replace=False)
        index_a.append(index[0])
        index_b.append(index[1])
    for i in index_a:
        i_a = np.random.randint(len(train_imgs_a[i]))
        train_feed_a.append(train_imgs_a[i][i_a])
    for i in index_b:
        i_b = np.random.randint(len(train_imgs_b[i]))
        train_feed_b.append(train_imgs_b[i][i_b])
    #########################################################    
    # labels for verification loss
    train_labels_vs_neg = np.zeros((int(batch_size/2), 2))
    # negative label: 01
    train_labels_vs_neg[:,1] = 1
    #########################################################    
    # labels for classification loss
    train_labels_cs_a_neg = np.zeros((int(batch_size/2), data_length))
    train_labels_cs_b_neg = np.zeros((int(batch_size/2), data_length))
    for i in range(train_labels_cs_a_neg.shape[0]):
        train_labels_cs_a_neg[i,index_a[i]] = 1
        train_labels_cs_b_neg[i,index_b[i]] = 1
    #########################################################    
    return np.array(train_feed_a), np.array(train_feed_b), train_labels_vs_neg, train_labels_cs_a_neg, train_labels_cs_b_neg

def next_batch_ft_single(train_imgs_a, train_imgs_b, batch_size, id_num):
    """Sampling a batch for training. This one is for single branch 
       network during fine tune stage, i.e. cnn-i, cnn-ic, cnn-frw-ic, cnn-fc-ic."""
    data_length = train_imgs_a.shape[0]
    images_feed_a = []
    index_a = np.random.choice(range(data_length), size=int(batch_size/2))
    for i in index_a:
        images_feed_a.append(train_imgs_a[i])
    images_feed_b = []
    index_b = np.random.choice(range(data_length), size=int(batch_size/2))
    for i in index_b:
        images_feed_b.append(train_imgs_b[i])
    # labels
    labels_feed_a = np.zeros((int(batch_size/2), id_num))
    for i in range(labels_feed_a.shape[0]):
        labels_feed_a[i,index_a[i]] = 1
    labels_feed_b = np.zeros((int(batch_size/2), id_num))
    for i in range(labels_feed_b.shape[0]):
        labels_feed_b[i,index_b[i]] = 1
    images_feed_a = np.array(images_feed_a)
    images_feed_b = np.array(images_feed_b)
    images_feed = np.concatenate([images_feed_a, images_feed_b], axis=0)
    labels_feed = np.concatenate([labels_feed_a, labels_feed_b], axis=0)
    return images_feed, labels_feed

def next_batch_ft_pair(train_imgs_a, train_imgs_b, batch_size, id_num):
    """Sampling a batch for training. This one is for two-branch 
       network during fine tune stage, i.e. cnn-iv."""
    imgs_pos_a, imgs_pos_b, labels_vs_pos, labels_cs_a_pos, labels_cs_b_pos = positive_batch_ft(train_imgs_a, train_imgs_b, batch_size, id_num)
    imgs_neg_a, imgs_neg_b, labels_vs_neg, labels_cs_a_neg, labels_cs_b_neg = negative_batch_ft(train_imgs_a, train_imgs_b, batch_size, id_num)
    imgs_a = np.concatenate([imgs_pos_a, imgs_neg_a], axis=0)
    imgs_b = np.concatenate([imgs_pos_b, imgs_neg_b], axis=0)
    labels_vs = np.concatenate([labels_vs_pos, labels_vs_neg], axis=0)
    labels_cs_a = np.concatenate([labels_cs_a_pos, labels_cs_a_neg], axis=0)
    labels_cs_b = np.concatenate([labels_cs_b_pos, labels_cs_b_neg], axis=0)
    return imgs_a, imgs_b, labels_vs, labels_cs_a, labels_cs_b

def positive_batch_ft(train_imgs_a, train_imgs_b, batch_size, id_num):
    """Sampling positive samples for two-branch network
       during fine tune stage."""
    train_feed_a = []
    train_feed_b = []
    data_length = train_imgs_a.shape[0]
    index = np.random.choice(range(data_length), size=int(batch_size/2))
    for i in index:
        train_feed_a.append(train_imgs_a[i])
        train_feed_b.append(train_imgs_b[i])
    #########################################################    
    # labels for verification loss
    train_labels_vs_pos = np.zeros((int(batch_size/2), 2))
    # positive label: 10
    train_labels_vs_pos[:,0] = 1
    #########################################################    
    # labels for classification loss
    train_labels_cs_a_pos = np.zeros((int(batch_size/2), id_num))
    train_labels_cs_b_pos = np.zeros((int(batch_size/2), id_num))
    for i in range(train_labels_cs_a_pos.shape[0]):
        train_labels_cs_a_pos[i,index[i]] = 1
        train_labels_cs_b_pos[i,index[i]] = 1
    #########################################################    
    return np.array(train_feed_a), np.array(train_feed_b), train_labels_vs_pos, train_labels_cs_a_pos, train_labels_cs_b_pos

def negative_batch_ft(train_imgs_a, train_imgs_b, batch_size, id_num):
    """Sampling negative samples for two-branch network
       during fine tune stage."""
    train_feed_a = []
    train_feed_b = []
    data_length = train_imgs_a.shape[0]
    index_a = []
    index_b = []
    for i in range(int(batch_size/2)):
        index = np.random.choice(range(data_length), size=2, replace=False)
        index_a.append(index[0])
        index_b.append(index[1])
    for i in index_a:
        train_feed_a.append(train_imgs_a[i])
    for i in index_b:
        train_feed_b.append(train_imgs_b[i])
    #########################################################    
    # labels for verification loss
    train_labels_vs_neg = np.zeros((int(batch_size/2), 2))
    # negative label: 01
    train_labels_vs_neg[:,1] = 1
    #########################################################    
    # labels for classification loss
    train_labels_cs_a_neg = np.zeros((int(batch_size/2), id_num))
    train_labels_cs_b_neg = np.zeros((int(batch_size/2), id_num))
    for i in range(train_labels_cs_a_neg.shape[0]):
        train_labels_cs_a_neg[i,index_a[i]] = 1
        train_labels_cs_b_neg[i,index_b[i]] = 1
    #########################################################    
    return np.array(train_feed_a), np.array(train_feed_b), train_labels_vs_neg, train_labels_cs_a_neg, train_labels_cs_b_neg

def labels_statistics(labels, batch_size):
    """Counting each label in a batch."""
    labels = list(np.argmax(labels,1))
    counter = Counter(labels)
    labels_stats = np.array([counter[l] for l in labels])
    labels_stats = labels_stats.reshape(batch_size, 1)
    return labels_stats

def eval(num_images, dist_sq):
    """Ranking accuracy evaluation.
       num_images: int, number of test images.
       dist_sq: [num_images,num_images], matrix of squared euclidean distance
       between image pairs of two cameras."""
    num_correct_rank1 = 0
    num_correct_rank5 = 0
    num_correct_rank10 = 0
    for i in range(num_images):
        index_sorted = sorted(range(num_images),key=lambda x:dist_sq[i,x])
        # rank 1
        index_sorted_1 = index_sorted[:1]
        if i in index_sorted_1:
            num_correct_rank1 += 1
        # rank 5
        index_sorted_5 = index_sorted[:5]
        if i in index_sorted_5:
            num_correct_rank5 += 1
        # rank 10
        index_sorted_10 = index_sorted[:10]
        if i in index_sorted_10:
            num_correct_rank10 += 1
    acc_rank1 = num_correct_rank1 / num_images
    acc_rank5 = num_correct_rank5 / num_images
    acc_rank10 = num_correct_rank10 / num_images
    return acc_rank1, acc_rank5, acc_rank10

def train(dataset, cnn_structure, current_seed=1, max_steps=25000, learning_rate=0.001, batch_size=100, alpha=0.001, landa=0.01, alfa=0.5, beta=0.001, C=200, save_path=None, restore_model=None):
    ################################################################
    print('---------------------------------------------------')
    print('dataset: %s' % dataset)
    print('cnn_structure: %s' % cnn_structure)
    print('seed: %d' % current_seed)
    print('max_steps: %d' % max_steps)
    print('learning_rate: %f' % learning_rate)
    print('batch_size: %d' % batch_size)
    # weight decay
    print('alpha: %f' % alpha)
    if cnn_structure == 'cnn-ic' or cnn_structure == 'cnn-fc-ic' or cnn_structure == 'cnn-frw-ic':
        # center loss importance
        print('landa: %f' % landa)
        # updating rate for centers
        print('alfa: %f' % alfa)
    if cnn_structure == 'cnn-frw-ic':
        # importance of FRW layer norm constraint
        print('beta: %f' % beta)
        # norm constraint constant for FRW layer 
        print('C: %d' % C)
    print('---------------------------------------------------')
    ################################################################
    if dataset == 'cuhk03':
        print('generating cuhk03...')
        cuhk03 = process_cuhk03.generate_cuhk03('../images/cuhk03/All_128x48/detected/', current_seed)
        # validation data is not used here
        data_train_a, data_train_b, _, _, data_test_a, data_test_b = cuhk03
        # randomly choose one as test data
        # fix numpy random seed
        np.random.seed(1)
        data_test_a = [x[np.random.randint(len(x))] for x in data_test_a]
        data_test_a = np.array(data_test_a)
        # fix numpy random seed
        np.random.seed(1)
        data_test_b = [x[np.random.randint(len(x))] for x in data_test_b]
        data_test_b = np.array(data_test_b)
    elif dataset == 'cuhk01':
        print('generating cuhk03 and market1501 for pretraining...')
        cumarket, train_data_mean = process_cuhk01.generate_cumarket('../images/cuhk03/All_128x48/detected/', '../images/Market1501/bounding_box_train_png/', '../images/Market1501/bounding_box_test_png/')
        data_train_a, data_train_b = cumarket
        print('generating cuhk01...')
        cuhk01 = process_cuhk01.generate_cuhk01('../images/cuhk01/cam1_resize/', '../images/cuhk01/cam2_resize/', train_data_mean, current_seed)
        data_ft_a, data_ft_b, data_test_a, data_test_b = cuhk01
        # randomly choose one as fine tune data
        # fix numpy random seed
        np.random.seed(1)
        data_ft_a = [x[np.random.randint(len(x))] for x in data_ft_a]
        data_ft_a = np.array(data_ft_a)
        # fix numpy random seed
        np.random.seed(1)
        data_ft_b = [x[np.random.randint(len(x))] for x in data_ft_b]
        data_ft_b = np.array(data_ft_b)
        # randomly choose one as test data
        # fix numpy random seed
        np.random.seed(1)
        data_test_a = [x[np.random.randint(len(x))] for x in data_test_a]
        data_test_a = np.array(data_test_a)
        # fix numpy random seed
        np.random.seed(1)
        data_test_b = [x[np.random.randint(len(x))] for x in data_test_b]
        data_test_b = np.array(data_test_b)
    elif dataset == 'viper':
        print('generating cuhk03 and market1501 for pretrain...')
        cumarket, train_data_mean = process_cuhk01.generate_cumarket('../images/cuhk03/All_128x48/detected/', '../images/Market1501/bounding_box_train_png/', '../images/Market1501/bounding_box_test_png/')
        data_train_a, data_train_b = cumarket
        print('generating viper...')
        viper = process_viper.generate_viper('../images/VIPeR/cam_a/', '../images/VIPeR/cam_b/', train_data_mean, current_seed)
        data_ft_a, data_ft_b, data_test_a, data_test_b = viper

    # number of identities of training data
    id_num = len(data_train_a)
    print('%d identities for training' % id_num) 

    if cnn_structure == 'cnn-i':
        images_placeholder = tf.placeholder(tf.float32, shape=[None, 128, 48, 3])
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, id_num))
        loss, h_fc1, w_fc2 = cnn_i(images_placeholder, labels_placeholder, id_num, alpha)
    elif cnn_structure == 'cnn-iv':
        images_placeholder_a = tf.placeholder(tf.float32, shape=[None, 128, 48, 3])
        images_placeholder_b = tf.placeholder(tf.float32, shape=[None, 128, 48, 3])
        labels_placeholder_vs = tf.placeholder(tf.float32, shape=(None, 2))
        labels_placeholder_cs_a = tf.placeholder(tf.float32, shape=(None, id_num))
        labels_placeholder_cs_b = tf.placeholder(tf.float32, shape=(None, id_num))
        loss, h_fc1_a, h_fc1_b, loss_cs, w_fc1_cs = cnn_iv(images_placeholder_a, images_placeholder_b, labels_placeholder_vs, labels_placeholder_cs_a, labels_placeholder_cs_b, id_num, alpha)
    elif cnn_structure == 'cnn-ic':
        images_placeholder = tf.placeholder(tf.float32, shape=[None, 128, 48, 3])
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, id_num))
        labels_placeholder_stats = tf.placeholder(tf.float32, shape=(None, 1))
        loss, h_fc1, centers, w_fc2 = cnn_ic(images_placeholder, labels_placeholder, labels_placeholder_stats, id_num, alpha, landa, alfa, batch_size)
    elif cnn_structure == 'cnn-fc-ic':
        images_placeholder = tf.placeholder(tf.float32, shape=[None, 128, 48, 3])
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, id_num))
        labels_placeholder_stats = tf.placeholder(tf.float32, shape=(None, 1))
        loss, h_fc2, centers, w_fc3 = cnn_fc_ic(images_placeholder, labels_placeholder, labels_placeholder_stats, id_num, alpha, landa, alfa, batch_size)
    elif cnn_structure == 'cnn-frw-ic':
        images_placeholder = tf.placeholder(tf.float32, shape=[None, 128, 48, 3])
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, id_num))
        labels_placeholder_stats = tf.placeholder(tf.float32, shape=(None, 1))
        loss, h_fc1, centers, w_fc2 = cnn_frw_ic(images_placeholder, labels_placeholder, labels_placeholder_stats, id_num, alpha, landa, alfa, beta, C, batch_size)

    global_step = tf.Variable(0, trainable=False)
    # decay learning rate
    decayed_learning_rate = tf.train.exponential_decay(learning_rate,global_step,22000,0.1,staircase=True)
    optimizer = tf.train.AdamOptimizer(decayed_learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # when there is no pretrained model
        if restore_model == None:
            sess.run(init)
            print('start training...')
            for step in range(max_steps+1):
                # sample training batch 
                if cnn_structure == 'cnn-i':
                    images_feed, labels_feed = next_batch_single(data_train_a, data_train_b, batch_size)
                    feed_dict = {images_placeholder: images_feed,
                                 labels_placeholder: labels_feed}
                elif cnn_structure == 'cnn-iv':
                    images_a_feed, images_b_feed, labels_feed_vs, labels_feed_cs_a, labels_feed_cs_b = next_batch_pair(data_train_a, data_train_b, batch_size)
                    feed_dict = {images_placeholder_a: images_a_feed,
                                 images_placeholder_b: images_b_feed,
                                 labels_placeholder_vs: labels_feed_vs,
                                 labels_placeholder_cs_a: labels_feed_cs_a,
                                 labels_placeholder_cs_b: labels_feed_cs_b}
                elif cnn_structure == 'cnn-ic' or cnn_structure == 'cnn-fc-ic' or cnn_structure == 'cnn-frw-ic':
                    images_feed, labels_feed = next_batch_single(data_train_a, data_train_b, batch_size)
                    labels_feed_stats = labels_statistics(labels_feed, batch_size)
                    feed_dict = {images_placeholder: images_feed,
                                 labels_placeholder: labels_feed,
                                 labels_placeholder_stats: labels_feed_stats}
                # update gradient
                if cnn_structure == 'cnn-i' or cnn_structure == 'cnn-iv':
                    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                else:
                    _, loss_value, centers_value = sess.run([train_op, loss, centers], feed_dict=feed_dict)
                if step % 100 == 0 and step != 0:
                    print('Step %d, loss = %.3f' % (step, loss_value))
            if save_path != None:
                saver.save(sess, save_path, global_step=global_step)
                print('Model saved in %s' % save_path)
        # when there is pretrained model
        else:
            saver.restore(sess, restore_model)
            print("Pretrained model restored.")
        if dataset == 'cuhk01' or dataset == 'viper':
            if cnn_structure == 'cnn-iv':
                train_cs = optimizer.minimize(loss_cs, var_list=[w_fc1_cs], global_step=global_step)
                # first stage of fine-tuning
                # only update the parameters of the last softmax
                print('Fine-tuning the last softmax...')
                for step in range(7001):
                    images_a_feed, images_b_feed, labels_feed_vs, labels_feed_cs_a, labels_feed_cs_b = next_batch_ft_pair(data_ft_a, data_ft_b, batch_size, id_num)
                    feed_dict = {images_placeholder_a: images_a_feed,
                                 images_placeholder_b: images_b_feed,
                                 labels_placeholder_vs: labels_feed_vs,
                                 labels_placeholder_cs_a: labels_feed_cs_a,
                                 labels_placeholder_cs_b: labels_feed_cs_b}
                    # update gradient
                    _, loss_cs_value = sess.run([train_cs, loss_cs], feed_dict=feed_dict)
                    if step%100==0:
                        print('Step %d, loss = %.3f' % (step, loss_cs_value))
                # second stage of fine-tuning
                # update all the parameters 
                print('Fine-tuning the whole network...')
                for step in range(101):
                    images_a_feed, images_b_feed, labels_feed_vs, labels_feed_cs_a, labels_feed_cs_b = next_batch_ft_pair(data_ft_a, data_ft_b, batch_size, id_num)
                    feed_dict = {images_placeholder_a: images_a_feed,
                                 images_placeholder_b: images_b_feed,
                                 labels_placeholder_vs: labels_feed_vs,
                                 labels_placeholder_cs_a: labels_feed_cs_a,
                                 labels_placeholder_cs_b: labels_feed_cs_b}
                    # update gradient
                    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                    if step%10 == 0:
                        print('Step %d, loss = %.3f' % (step, loss_value))
            elif cnn_structure == 'cnn-i':
                train_sm = optimizer.minimize(loss, var_list=[w_fc2], global_step=global_step)
                # first stage of fine-tuning
                # only update the parameters of the last softmax
                print('Fine-tuning the last softmax...')
                for step in range(7001):
                    images_feed, labels_feed = next_batch_ft_single(data_ft_a, data_ft_b, batch_size, id_num)
                    feed_dict = {images_placeholder: images_feed,
                                 labels_placeholder: labels_feed}
                    # update gradient
                    _, loss_value = sess.run([train_sm, loss], feed_dict=feed_dict)
                    if step % 100 == 0:
                        print('Step %d, loss = %.3f' % (step, loss_value))
                # second stage of fine-tuning
                # update all the parameters 
                print('Fine-tuning the whole network...')
                for step in range(101):
                    images_feed, labels_feed = next_batch_ft_single(data_ft_a, data_ft_b, batch_size, id_num)
                    feed_dict = {images_placeholder: images_feed,
                                 labels_placeholder: labels_feed}
                    # update gradient
                    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                    if step % 10 == 0:
                        print('Step %d, loss = %.3f' % (step, loss_value))
            elif cnn_structure == 'cnn-ic' or cnn_structure == 'cnn-frw-ic':
                train_sm = optimizer.minimize(loss, var_list=[w_fc2], global_step=global_step)
                # first stage of fine-tuning
                # only update the parameters of the last softmax
                print('Fine-tuning the last softmax...')
                for step in range(7001):
                    images_feed, labels_feed = next_batch_ft_single(data_ft_a, data_ft_b, batch_size, id_num)
                    labels_feed_stats = labels_statistics(labels_feed, batch_size)
                    feed_dict = {images_placeholder: images_feed,
                                 labels_placeholder: labels_feed,
                                 labels_placeholder_stats: labels_feed_stats}
                    # update gradient
                    _, loss_value, centers_value = sess.run([train_sm, loss, centers], feed_dict=feed_dict)
                    if step % 100 == 0:
                        print('Step %d, loss = %.3f' % (step, loss_value))
                # second stage of fine-tuning
                # update all the parameters 
                print('Fine-tuning the whole network...')
                for step in range(101):
                    images_feed, labels_feed = next_batch_ft_single(data_ft_a, data_ft_b, batch_size, id_num)
                    labels_feed_stats = labels_statistics(labels_feed, batch_size)
                    feed_dict = {images_placeholder: images_feed,
                                 labels_placeholder: labels_feed,
                                 labels_placeholder_stats: labels_feed_stats}
                    # update gradient
                    _, loss_value, centers_value = sess.run([train_op, loss, centers], feed_dict=feed_dict)
                    if step % 10 == 0:
                        print('Step %d, loss = %.3f' % (step, loss_value))
            else:
                train_sm = optimizer.minimize(loss, var_list=[w_fc3], global_step=global_step)
                # first stage of fine-tuning
                # only update the parameters of the last softmax
                print('Fine-tuning the last softmax...')
                for step in range(7001):
                    images_feed, labels_feed = next_batch_ft_single(data_ft_a, data_ft_b, batch_size, id_num)
                    labels_feed_stats = labels_statistics(labels_feed, batch_size)
                    feed_dict = {images_placeholder: images_feed,
                                 labels_placeholder: labels_feed,
                                 labels_placeholder_stats: labels_feed_stats}
                    # update gradient
                    _, loss_value, centers_value = sess.run([train_sm, loss, centers], feed_dict=feed_dict)
                    if step % 100 == 0:
                        print('Step %d, loss = %.3f' % (step, loss_value))
                # second stage of fine-tuning
                # update all the parameters 
                print('Fine-tuning the whole network...')
                for step in range(101):
                    images_feed, labels_feed = next_batch_ft_single(data_ft_a, data_ft_b, batch_size, id_num)
                    labels_feed_stats = labels_statistics(labels_feed, batch_size)
                    feed_dict = {images_placeholder: images_feed,
                                 labels_placeholder: labels_feed,
                                 labels_placeholder_stats: labels_feed_stats}
                    # update gradient
                    _, loss_value, centers_value = sess.run([train_op, loss, centers], feed_dict=feed_dict)
                    if step % 10 == 0:
                        print('Step %d, loss = %.3f' % (step, loss_value))
        print('start testing...')
        # as for batch norm, it computes statistics directly from testing batches 
        if cnn_structure == 'cnn-i':
            images_feed = data_test_a
            # labels does not matter
            labels_feed = np.zeros((data_test_a.shape[0], id_num))
            feed_dict = {images_placeholder: images_feed,
                         labels_placeholder: labels_feed}
            h_fc1_value_a = sess.run(h_fc1, feed_dict=feed_dict)
            images_feed = data_test_b
            # labels does not matter
            labels_feed = np.zeros((data_test_b.shape[0], id_num))
            feed_dict = {images_placeholder: images_feed,
                         labels_placeholder: labels_feed}
            h_fc1_value_b = sess.run(h_fc1, feed_dict=feed_dict)
        elif cnn_structure == 'cnn-iv':
            images_a_feed = data_test_a
            images_b_feed = data_test_b
            # labels does not matter
            labels_feed_vs = np.zeros((data_test_a.shape[0], 2))
            labels_feed_cs_a = np.zeros((data_test_a.shape[0], id_num))
            labels_feed_cs_b = np.zeros((data_test_b.shape[0], id_num))
            feed_dict = {images_placeholder_a: images_a_feed,
                         images_placeholder_b: images_b_feed,
                         labels_placeholder_vs: labels_feed_vs,
                         labels_placeholder_cs_a: labels_feed_cs_a,
                         labels_placeholder_cs_b: labels_feed_cs_b}
            h_fc1_value_a, h_fc1_value_b = sess.run([h_fc1_a, h_fc1_b], feed_dict=feed_dict)
        elif cnn_structure == 'cnn-ic' or cnn_structure == 'cnn-frw-ic':
            images_feed = data_test_a
            # labels does not matter
            labels_feed = np.zeros((data_test_a.shape[0], id_num))
            labels_feed_stats = np.zeros((data_test_a.shape[0], 1))
            feed_dict = {images_placeholder: images_feed,
                         labels_placeholder: labels_feed,
                         labels_placeholder_stats: labels_feed_stats}
            h_fc1_value_a = sess.run(h_fc1, feed_dict=feed_dict)
            images_feed = data_test_b
            # labels does not matter
            labels_feed = np.zeros((data_test_b.shape[0], id_num))
            labels_feed_stats = np.zeros((data_test_b.shape[0], 1))
            feed_dict = {images_placeholder: images_feed,
                         labels_placeholder: labels_feed,
                         labels_placeholder_stats: labels_feed_stats}
            h_fc1_value_b = sess.run(h_fc1, feed_dict=feed_dict)
        elif cnn_structure == 'cnn-fc-ic':
            images_feed = data_test_a
            # labels does not matter
            labels_feed = np.zeros((data_test_a.shape[0], id_num))
            labels_feed_stats = np.zeros((data_test_a.shape[0], 1))
            feed_dict = {images_placeholder: images_feed,
                         labels_placeholder: labels_feed,
                         labels_placeholder_stats: labels_feed_stats}
            h_fc1_value_a = sess.run(h_fc2, feed_dict=feed_dict)
            images_feed = data_test_b
            # labels does not matter
            labels_feed = np.zeros((data_test_b.shape[0], id_num))
            labels_feed_stats = np.zeros((data_test_b.shape[0], 1))
            feed_dict = {images_placeholder: images_feed,
                         labels_placeholder: labels_feed,
                         labels_placeholder_stats: labels_feed_stats}
            h_fc1_value_b = sess.run(h_fc2, feed_dict=feed_dict)

        # normalize each feature vector to a unit norm
        h_fc1_value_a_unit = h_fc1_value_a / np.sqrt(np.sum(np.square(h_fc1_value_a),axis=1).reshape(-1,1))
        h_fc1_value_b_unit = h_fc1_value_b / np.sqrt(np.sum(np.square(h_fc1_value_b),axis=1).reshape(-1,1))
        distance_mat = []
        for i in range(h_fc1_value_a_unit.shape[0]):
            distance = np.sum(np.square(h_fc1_value_a_unit[i,:]-h_fc1_value_b_unit),1).reshape(1,h_fc1_value_b_unit.shape[0])
            distance_mat.append(distance)
        distance_mat = np.concatenate(distance_mat, axis=0)
        acc_rank1, acc_rank5, acc_rank10 = eval(distance_mat.shape[0], distance_mat)
        print('Test Accuracy(single-shot), rank1: %.3f, rank5: %.3f, rank10: %.3f' % (acc_rank1, acc_rank5, acc_rank10))

