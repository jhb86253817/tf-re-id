# preprocessing for cuhk03(detected) and market1501, as pretraining data
# preprocessing for cuhk01, as fine tuning data and testing data
# cuhk01 has 971 identities in total, 485 for training, 486 for testing
# transform raw images to numpy arrays
import numpy as np
import os
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

def generate_cumarket(cuhk03_file, market_train_file, market_test_file):
    """The combination of cuhk03 and market1501, for pretraining."""
    # cuhk03
    print('processing cuhk03...')
    # read names of images
    images_list = os.listdir(cuhk03_file)
    images_list = sorted(images_list)
    # remove images of batch 4 and 5, as most other works do
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

    # flipping images
    cam_a_nested_flip = []
    for l in cam_a_nested:
        cam_a_nested_flip.append([])
        for p in l:
            cam_a_nested_flip[-1].append(flip(p))
    
    cam_b_nested_flip = []
    for l in cam_b_nested:
        cam_b_nested_flip.append([])
        for p in l:
            cam_b_nested_flip[-1].append(flip(p))

    # translational transforms, 3 samples for each image
    cam_a_nested_trans = []
    for l in cam_a_nested:
        cam_a_nested_trans.append([])
        for p in l:
            for i in range(3):
                cam_a_nested_trans[-1].append(trans(p, 6, 2))
    
    cam_b_nested_trans = []
    for l in cam_b_nested:
        cam_b_nested_trans.append([])
        for p in l:
            for i in range(3):
                cam_b_nested_trans[-1].append(trans(p, 6, 2))

    # combine the lists
    for i in range(len(cam_a_nested)):
        cam_a_nested[i] = cam_a_nested[i] + cam_a_nested_flip[i] + cam_a_nested_trans[i]
    for i in range(len(cam_b_nested)):
        cam_b_nested[i] = cam_b_nested[i] + cam_b_nested_flip[i] + cam_b_nested_trans[i]
    
    for i in range(len(name_a_nested)):
        name_a_nested[i] = name_a_nested[i] * 5
    for i in range(len(name_b_nested)):
        name_b_nested[i] = name_b_nested[i] * 5

    # market1501
    print('processing market1501...')
    images_list_train = os.listdir(market_train_file)
    images_list_test = os.listdir(market_test_file)
    images_list_train = sorted(images_list_train)
    images_list_test = sorted(images_list_test)
    # remove noisy images
    images_list_test = [x for x in images_list_test if x[:2]!='-1' and x[:4]!='0000']

    cam_1 = []
    name_1 = []
    cam_2 = []
    name_2 = []
    # split the original market1501 6 cameras into 2 cameras
    # camera 1,2,3 as a group, camera 4,5,6 as the other
    for image_path in images_list_train:
        image = plt.imread(market_train_file+image_path)
        if image_path[6] == '1' or image_path[6] == '2' or image_path[6] == '3':
            name_1.append(image_path)
            cam_1.append(image)
        elif image_path[6] == '4' or image_path[6] == '5' or image_path[6] == '6': 
            name_2.append(image_path)
            cam_2.append(image)
    for image_path in images_list_test:
        image = plt.imread(market_test_file+image_path)
        if image_path[6] == '1' or image_path[6] == '2' or image_path[6] == '3':
            name_1.append(image_path)
            cam_1.append(image)
        elif image_path[6] == '4' or image_path[6] == '5' or image_path[6] == '6': 
            name_2.append(image_path)
            cam_2.append(image)
    
    # for each camera list, further group images of the same idetity in one list
    name_1_set = set([x[:4] for x in name_1])
    name_2_set = set([x[:4] for x in name_2])
    # cam_1
    name_1_nested = []
    cam_1_nested = []
    current_name = 'xxxxxx'
    for i in range(len(name_1)):
        if name_1[i][:4] != current_name:
            if name_1[i][:4] in name_1_set and name_1[i][:4] in name_2_set:
                name_1_nested.append([])
                cam_1_nested.append([])
                current_name = name_1[i][:4]
            else:
                continue
        name_1_nested[-1].append(name_1[i])
        cam_1_nested[-1].append(cam_1[i])
    # cam_2
    name_2_nested = []
    cam_2_nested = []
    current_name = 'xxxxxx'
    for i in range(len(name_2)):
        if name_2[i][:4] != current_name:
            if name_2[i][:4] in name_1_set and name_2[i][:4] in name_2_set:
                name_2_nested.append([])
                cam_2_nested.append([])
                current_name = name_2[i][:4]
            else:
                continue
        name_2_nested[-1].append(name_2[i])
        cam_2_nested[-1].append(cam_2[i])
    
    # flipping images
    cam_1_nested_flip = []
    for l in cam_1_nested:
        cam_1_nested_flip.append([])
        for p in l:
            cam_1_nested_flip[-1].append(flip(p))
    
    cam_2_nested_flip = []
    for l in cam_2_nested:
        cam_2_nested_flip.append([])
        for p in l:
            cam_2_nested_flip[-1].append(flip(p))
    
    # translational transforms, 3 samples for each image
    cam_1_nested_trans = []
    for l in cam_1_nested:
        cam_1_nested_trans.append([])
        for p in l:
            for i in range(3):
                cam_1_nested_trans[-1].append(trans(p, 6, 2))
    
    cam_2_nested_trans = []
    for l in cam_2_nested:
        cam_2_nested_trans.append([])
        for p in l:
            for i in range(3):
                cam_2_nested_trans[-1].append(trans(p, 6, 2))
    
    # combine the lists
    for i in range(len(cam_1_nested)):
        cam_1_nested[i] = cam_1_nested[i] + cam_1_nested_flip[i] + cam_1_nested_trans[i]
    for i in range(len(cam_2_nested)):
        cam_2_nested[i] = cam_2_nested[i] + cam_2_nested_flip[i] + cam_2_nested_trans[i]
    
    for i in range(len(name_1_nested)):
        name_1_nested[i] = name_1_nested[i] * 5
    for i in range(len(name_2_nested)):
        name_2_nested[i] = name_2_nested[i] * 5

    print('combine cuhk03 and market1501...')
    # combining cuhk03 and market1501
    cam_a_nested = cam_a_nested + cam_1_nested
    name_a_nested = name_a_nested + name_1_nested
    cam_b_nested = cam_b_nested + cam_2_nested
    name_b_nested = name_b_nested + name_2_nested

    # subtracting mean
    cam_a_nested = [x for l in cam_a_nested for x in l]
    cam_a_nested = np.array(cam_a_nested)
    cam_a_nested = cam_a_nested.reshape(-1,128*48*3)
    cam_b_nested = [x for l in cam_b_nested for x in l]
    cam_b_nested = np.array(cam_b_nested)
    cam_b_nested = cam_b_nested.reshape(-1,128*48*3)
    # compute mean from both cam_a and cam_b
    temp_mean = np.mean(np.concatenate([cam_a_nested,cam_b_nested],axis=0), axis=0)
    # subtract mean for cam_a 
    cam_a_nested -= temp_mean
    cam_a_nested = cam_a_nested.reshape(-1,128,48,3)
    # subtract mean for cam_b 
    cam_b_nested -= temp_mean
    cam_b_nested = cam_b_nested.reshape(-1,128,48,3)
    
    # get the nested list again for cam_a 
    cam_a_nested_mean = []
    j = 0
    for i in range(len(name_a_nested)):
        cam_a_nested_mean.append([])
        until = j+len(name_a_nested[i])
        while j < until:
            cam_a_nested_mean[-1].append(cam_a_nested[j,:])
            j += 1
    
    # get the nested list again for cam_b 
    cam_b_nested_mean = []
    j = 0
    for i in range(len(name_b_nested)):
        cam_b_nested_mean.append([])
        until = j+len(name_b_nested[i])
        while j < until:
            cam_b_nested_mean[-1].append(cam_b_nested[j,:])
            j += 1
    cumarket = (cam_a_nested_mean, cam_b_nested_mean)
    return cumarket, temp_mean

def generate_cuhk01(cuhk01_a_file, cuhk01_b_file, train_data_mean, current_seed):
    # cuhk01
    print('processing cuhk01...')
    cam_a = os.listdir(cuhk01_a_file)
    cam_b = os.listdir(cuhk01_b_file)
    cam_a = sorted(cam_a)
    cam_b = sorted(cam_b)
    
    images_a = []
    images_b = []
    for image_path in cam_a:
        image = plt.imread(cuhk01_a_file+image_path)
        images_a.append(image)
    for image_path in cam_b:
        image = plt.imread(cuhk01_b_file+image_path)
        images_b.append(image)
    
    images_a = np.array(images_a)
    images_b = np.array(images_b)
    images_a = images_a.reshape(-1,128*48*3)
    images_b = images_b.reshape(-1,128*48*3)
    # subtract mean
    images_a = images_a - train_data_mean
    images_b = images_b - train_data_mean
    
    images_a = images_a.reshape(-1,128,48,3)
    images_b = images_b.reshape(-1,128,48,3)
    
    images_a_nested = []
    images_b_nested = []
    for i in range(images_a.shape[0]):
        if i%2==0:
            images_a_nested.append([])
        images_a_nested[-1].append(images_a[i])
    for i in range(images_b.shape[0]):
        if i%2==0:
            images_b_nested.append([])
        images_b_nested[-1].append(images_b[i])
    
    # randomly select 485 as train data, other 486 as test data
    np.random.seed(current_seed)
    train_index = np.random.choice(range(len(images_a_nested)), size=485, replace=False)
    test_index = [x for x in range(len(images_a_nested)) if x not in train_index]
    images_a_nested_train = [images_a_nested[i] for i in range(len(images_a_nested)) if i in train_index]
    images_b_nested_train = [images_b_nested[i] for i in range(len(images_b_nested)) if i in train_index]
    images_a_nested_test = [images_a_nested[i] for i in range(len(images_a_nested)) if i in test_index]
    images_b_nested_test = [images_b_nested[i] for i in range(len(images_b_nested)) if i in test_index]
    cuhk01 = (images_a_nested_train, images_b_nested_train, images_a_nested_test, images_b_nested_test)
    return cuhk01
