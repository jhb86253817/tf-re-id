# Person Re-Identification in TensorFlow

This is code of the paper ["Deep Person Re-Identification with Improved Embedding and Efficient Training"](https://arxiv.org/abs/1705.03332). 

## Datasets
- [cuhk03](http://www.ee.cuhk.edu.hk/~rzhao/) 
- [cuhk01](http://www.ee.cuhk.edu.hk/~rzhao/)
- [viper](https://vision.soe.ucsc.edu/node/178)
- [market1501](http://www.liangzheng.org/Project/project_reid.html)(for pre-training, v15.09.15 is used in this paper)

## Environment
- Ubuntu or Windows
- TensorFlow v1.0
- Python 2.7 or 3.5
- Numpy, Matplotlib, Pillow

## Train Model
1. git clone the repo
2. Make a directory named "images" under "tf-re-id". Download the datasets, and put them under "images". For cuhk03, the directory name should be "cuhk03", its subfolder should be "All_128x48" where it contains two folders "detected" and "labeled". For cuhk01, the directory name should be "cuhk01", and it contains two folders "cam1" and "cam2". For viper, the directory name should be "VIPeR", and it contains two folders "cam_a" and "cam_b". For market1501, its folder name should be "Market1501", and it should have three subfolders "bounding_box_train", "bounding_box_test", and "query".
3. Run ``python preprocess_cuhk01.py`` to resize cuhk01 data to 128 x 48. Run ``python preprocess_market.py`` to resize market1501 data to 128 x 48. Other datasets are 128 x 48 originally.
4. Run ``main.py`` to start training. You need to specify arguments for training, which includes the name of the data, the CNN architecture, the seed of running. For example, if you want to run the CNN with center loss and FRW layer on cuhk03 in seed 1, then ``python main.py -cuhk03 -cnn-ic -1``. For more details, please see the usage information in ``main.py``.

Note that although the random seeds have been fixed, the results are not exactly reproducible because there is still randomness from GPU. However, the randomness is quite small.
