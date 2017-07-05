# market1501, convert jpg to png, and resize them to 128 x 48
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# training data
# read names of images
images_list_train = os.listdir('../images/Market1501/bounding_box_train')
images_list_train = sorted(images_list_train)
# remove meaningless files
images_list_train = [f for f in images_list_train if '.jpg' in f]
print(len(images_list_train))

if not os.path.exists('../images/Market1501/bounding_box_train_png'):
    os.makedirs('../images/Market1501/bounding_box_train_png')
if not os.path.exists('../images/Market1501/bounding_box_test_png'):
    os.makedirs('../images/Market1501/bounding_box_test_png')
if not os.path.exists('../images/Market1501/query_png'):
    os.makedirs('../images/Market1501/query_png')

for image_path in images_list_train:
    image = Image.open('../images/Market1501/bounding_box_train/'+image_path)
    image = image.resize((48,128), Image.ANTIALIAS)
    image.save('../images/Market1501/bounding_box_train_png/'+image_path[:-3]+'png')

# test data
# read names of images
images_list_test = os.listdir('../images/Market1501/bounding_box_test')
images_list_test = sorted(images_list_test)
# remove meaningless files
images_list_test = [f for f in images_list_test if '.jpg' in f]
# files start with '-1' are junks
images_list_test = [x for x in images_list_test if x[:2]!='-1']
# files start with '0000' are distractors
images_list_test = [x for x in images_list_test if x[:4]!='0000']
print(len(images_list_test))

for image_path in images_list_test:
    image = Image.open('../images/Market1501/bounding_box_test/'+image_path)
    image = image.resize((48,128), Image.ANTIALIAS)
    image.save('../images/Market1501/bounding_box_test_png/'+image_path[:-3]+'png')

# query
# read names of images
images_list_query = os.listdir('../images/Market1501/query')
images_list_query = sorted(images_list_query)
# remove meaningless files
images_list_query = [f for f in images_list_query if '.jpg' in f]
print(len(images_list_query))

for image_path in images_list_query:
    image = Image.open('../images/Market1501/query/'+image_path)
    image = image.resize((48,128), Image.ANTIALIAS)
    image.save('../images/Market1501/query_png/'+image_path[:-3]+'png')

