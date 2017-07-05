# preprocessing for cuhk03(detected)
# transform raw images to numpy arrays
# 1360 identities in total, 1160 for training, 100 for testing,
# 100 for validation (not used in this work)
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def flip(img_ori):
    """Horizontally flipping image"""
    img = np.copy(img_ori)
    return img[:,::-1,:]

def trans(img_ori, range_h=6, range_w=2):
    """Do translational transforms for train images.
       range_h: int, pixels of random perturbation on height.
       range_w: int, pixels of random perturbation on width.
    """
    img = np.copy(img_ori)

    rand_h = np.random.randint(-range_h, range_h+1)
    rand_w = np.random.randint(-range_w, range_w+1)
    
    if rand_h > 0:
        img[:rand_h,:] = 0.0
        img = np.concatenate([img[rand_h:,:], img[:rand_h,:]], axis=0)
    elif rand_h < 0:
        img[rand_h:,:] = 0.0
        img = np.concatenate([img[rand_h:,:], img[:rand_h,:]], axis=0)
    
    if rand_w > 0:
        img[:,:rand_w,:] = 0.0
        img = np.concatenate([img[:,rand_w:,:], img[:,:rand_w,:]], axis=1)
    elif rand_w < 0:
        img[:,rand_w:,:] = 0.0
        img = np.concatenate([img[:,rand_w:,:], img[:,:rand_w,:]], axis=1)

    return img

def generate_cuhk03(cuhk03_file, current_seed):
    # 20 test set random splits from the original data
    test_index_all = pickle.load(open('../utils/cuhk03_testsets', 'rb'))
    test_index = []
    # x indicates the group of the data, y is the id
    for x,y in test_index_all[current_seed-1]:
        if x == 0:
            test_index.append(int(y))
        elif x == 1:
            test_index.append(int(843+y))
        elif x == 2:
            test_index.append(int(1283+y))

    # generate random 100 identities as validation data
    np.random.seed(current_seed)
    train_valid_index = [x for x in range(1360) if x not in test_index]
    valid_index = np.random.choice(train_valid_index, size=100, replace=False)
    train_index = [x for x in train_valid_index if x not in valid_index]

    # read names of images
    images_list = os.listdir(cuhk03_file)
    images_list = sorted(images_list)
    # remove images of batch 6 and 5, as most other works do
    images_list = [x for x in images_list if x[0]!='4' and x[0]!='5']
    
    name_a = []
    cam_a = []
    name_b = []
    cam_b = []

    # read images, and store them in two lists as for two cameras
    for image_path in images_list:
        image = plt.imread(cuhk03_file+image_path)
        if image_path[6] == '1':
            name_a.append(image_path)
            cam_a.append(image)
        else:
            name_b.append(image_path)
            cam_b.append(image)

    # for each camera list, further group images of the same idetity in one list
    # cam_a
    name_a_nested = []
    cam_a_nested = []
    current_name = 'xxxxxx'
    for i in range(len(name_a)):
        if name_a[i][:5] != current_name:
            name_a_nested.append([])
            cam_a_nested.append([])
            current_name = name_a[i][:5]
        name_a_nested[-1].append(name_a[i])
        cam_a_nested[-1].append(cam_a[i])
    
    # for each camera list, further group images of the same idetity in one list
    # cam_b
    name_b_nested = []
    cam_b_nested = []
    current_name = 'xxxxxx'
    for i in range(len(name_b)):
        if name_b[i][:5] != current_name:
            name_b_nested.append([])
            cam_b_nested.append([])
            current_name = name_b[i][:5]
        name_b_nested[-1].append(name_b[i])
        cam_b_nested[-1].append(cam_b[i])

    name_a_nested_train = [name_a_nested[i] for i in range(len(name_a_nested)) if i in train_index]
    cam_a_nested_train = [cam_a_nested[i] for i in range(len(cam_a_nested)) if i in train_index]
    name_a_nested_test = [name_a_nested[i] for i in test_index]
    cam_a_nested_test = [cam_a_nested[i] for i in test_index]
    name_a_nested_valid = [name_a_nested[i] for i in valid_index]
    cam_a_nested_valid = [cam_a_nested[i] for i in valid_index]
    
    name_b_nested_train = [name_b_nested[i] for i in range(len(name_b_nested)) if i in train_index]
    cam_b_nested_train = [cam_b_nested[i] for i in range(len(cam_b_nested)) if i in train_index]
    name_b_nested_test = [name_b_nested[i] for i in test_index]
    cam_b_nested_test = [cam_b_nested[i] for i in test_index]
    name_b_nested_valid = [name_b_nested[i] for i in valid_index]
    cam_b_nested_valid = [cam_b_nested[i] for i in valid_index]

    # flipping images
    cam_a_nested_train_flip = []
    for l in cam_a_nested_train:
        cam_a_nested_train_flip.append([])
        for p in l:
            cam_a_nested_train_flip[-1].append(flip(p))
    
    cam_b_nested_train_flip = []
    for l in cam_b_nested_train:
        cam_b_nested_train_flip.append([])
        for p in l:
            cam_b_nested_train_flip[-1].append(flip(p))

    # translational transforms, 3 samples for each image 
    cam_a_nested_train_trans = []
    for l in cam_a_nested_train:
        cam_a_nested_train_trans.append([])
        for p in l:
            for i in range(3):
                cam_a_nested_train_trans[-1].append(trans(p, 6, 2))
    
    cam_b_nested_train_trans = []
    for l in cam_b_nested_train:
        cam_b_nested_train_trans.append([])
        for p in l:
            for i in range(3):
                cam_b_nested_train_trans[-1].append(trans(p, 6, 2))

    # combine the lists
    for i in range(len(cam_a_nested_train)):
        cam_a_nested_train[i] = cam_a_nested_train[i] + cam_a_nested_train_flip[i] + cam_a_nested_train_trans[i]
    for i in range(len(cam_b_nested_train)):
        cam_b_nested_train[i] = cam_b_nested_train[i] + cam_b_nested_train_flip[i] + cam_b_nested_train_trans[i]
    
    for i in range(len(name_a_nested_train)):
        name_a_nested_train[i] = name_a_nested_train[i] * 5
    for i in range(len(name_b_nested_train)):
        name_b_nested_train[i] = name_b_nested_train[i] * 5

    # subtracting mean for train data
    cam_a_nested_train = [x for l in cam_a_nested_train for x in l]
    cam_a_nested_train = np.array(cam_a_nested_train)
    cam_a_nested_train = cam_a_nested_train.reshape(-1,128*48*3)
    cam_b_nested_train = [x for l in cam_b_nested_train for x in l]
    cam_b_nested_train = np.array(cam_b_nested_train)
    cam_b_nested_train = cam_b_nested_train.reshape(-1,128*48*3)
    # compute mean from both cam_a and cam_b
    temp_mean = np.mean(np.concatenate([cam_a_nested_train,cam_b_nested_train],axis=0), axis=0)
    # subtract mean for cam_a 
    cam_a_nested_train -= temp_mean
    cam_a_nested_train = cam_a_nested_train.reshape(-1,128,48,3)
    # subtract mean for cam_b 
    cam_b_nested_train -= temp_mean
    cam_b_nested_train = cam_b_nested_train.reshape(-1,128,48,3)
    
    # get the nested list again for cam_a train
    cam_a_nested_train_mean = []
    j = 0
    for i in range(len(name_a_nested_train)):
        cam_a_nested_train_mean.append([])
        until = j+len(name_a_nested_train[i])
        while j < until:
            cam_a_nested_train_mean[-1].append(cam_a_nested_train[j,:])
            j += 1
    
    # get the nested list again for cam_b train
    cam_b_nested_train_mean = []
    j = 0
    for i in range(len(name_b_nested_train)):
        cam_b_nested_train_mean.append([])
        until = j+len(name_b_nested_train[i])
        while j < until:
            cam_b_nested_train_mean[-1].append(cam_b_nested_train[j,:])
            j += 1

    # subtracting mean for test data
    # subtract mean for cam_a test
    cam_a_nested_test = [x for l in cam_a_nested_test for x in l]
    cam_a_nested_test = np.array(cam_a_nested_test)
    cam_a_nested_test = cam_a_nested_test.reshape(-1,128*48*3)
    cam_a_nested_test -= temp_mean
    cam_a_nested_test = cam_a_nested_test.reshape(-1,128,48,3)
    
    # get the nested list again for cam_a test
    cam_a_nested_test_mean = []
    j = 0
    for i in range(len(name_a_nested_test)):
        cam_a_nested_test_mean.append([])
        until = j+len(name_a_nested_test[i])
        while j < until:
            cam_a_nested_test_mean[-1].append(cam_a_nested_test[j,:])
            j += 1
    
    # subtract mean for cam_b test
    cam_b_nested_test = [x for l in cam_b_nested_test for x in l]
    cam_b_nested_test = np.array(cam_b_nested_test)
    cam_b_nested_test = cam_b_nested_test.reshape(-1,128*48*3)
    cam_b_nested_test -= temp_mean
    cam_b_nested_test = cam_b_nested_test.reshape(-1,128,48,3)
    
    # get the nested list again for cam_b test
    cam_b_nested_test_mean = []
    j = 0
    for i in range(len(name_b_nested_test)):
        cam_b_nested_test_mean.append([])
        until = j+len(name_b_nested_test[i])
        while j < until:
            cam_b_nested_test_mean[-1].append(cam_b_nested_test[j,:])
            j += 1

    # subtracting mean for valid data
    # subtract mean for cam_a valid
    cam_a_nested_valid = [x for l in cam_a_nested_valid for x in l]
    cam_a_nested_valid = np.array(cam_a_nested_valid)
    cam_a_nested_valid = cam_a_nested_valid.reshape(-1,128*48*3)
    cam_a_nested_valid -= temp_mean
    cam_a_nested_valid = cam_a_nested_valid.reshape(-1,128,48,3)
    
    # get the nested list again for cam_a valid
    cam_a_nested_valid_mean = []
    j = 0
    for i in range(len(name_a_nested_valid)):
        cam_a_nested_valid_mean.append([])
        until = j+len(name_a_nested_valid[i])
        while j < until:
            cam_a_nested_valid_mean[-1].append(cam_a_nested_valid[j,:])
            j += 1
    
    # subtract mean for cam_b valid
    cam_b_nested_valid = [x for l in cam_b_nested_valid for x in l]
    cam_b_nested_valid = np.array(cam_b_nested_valid)
    cam_b_nested_valid = cam_b_nested_valid.reshape(-1,128*48*3)
    cam_b_nested_valid -= temp_mean
    cam_b_nested_valid = cam_b_nested_valid.reshape(-1,128,48,3)
    
    # get the nested list again for cam_b valid
    cam_b_nested_valid_mean = []
    j = 0
    for i in range(len(name_b_nested_valid)):
        cam_b_nested_valid_mean.append([])
        until = j+len(name_b_nested_valid[i])
        while j < until:
            cam_b_nested_valid_mean[-1].append(cam_b_nested_valid[j,:])
            j += 1
    cuhk03 = (cam_a_nested_train_mean, cam_b_nested_train_mean, cam_a_nested_valid_mean, cam_b_nested_valid_mean, cam_a_nested_test_mean, cam_b_nested_test_mean)
    return cuhk03
