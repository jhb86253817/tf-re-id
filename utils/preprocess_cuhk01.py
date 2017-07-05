# cuhk01, resize to 128 x 48
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

cam1 = os.listdir('../images/cuhk01/cam1')
cam1 = [f for f in cam1 if '.png' in f]
cam1 = sorted(cam1)

cam2 = os.listdir('../images/cuhk01/cam2')
cam2 = [f for f in cam2 if '.png' in f]
cam2 = sorted(cam2)

if not os.path.exists('../images/cuhk01/cam1_resize'):
    os.makedirs('../images/cuhk01/cam1_resize')
if not os.path.exists('../images/cuhk01/cam2_resize'):
    os.makedirs('../images/cuhk01/cam2_resize')

for f in cam1:
    image = Image.open('../images/cuhk01/cam1/'+f)
    image = image.resize((48,128), Image.ANTIALIAS)
    image.save('../images/cuhk01/cam1_resize/'+f)

for f in cam2:
    image = Image.open('../images/cuhk01/cam2/'+f)
    image = image.resize((48,128), Image.ANTIALIAS)
    image.save('../images/cuhk01/cam2_resize/'+f)

