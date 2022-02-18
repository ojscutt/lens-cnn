# ======================================================================================
# IMPORT RELEVANT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import cv2
from natsort import natsorted

# ======================================================================================
# LOAD IMAGES FROM TRAIN/TEST FILE

## define train or test set
im_set = 2 # 1 = train, 2 = test

## define resolution scale (use same value as in image simulation)
res_scale = 1 #set desired resolution scaling factor (60x60=1, 30x30=0.5 etc.)

img_dir = os.path.join(os.path.realpath('.'), 'dataset' + str(im_set) + '/images') 
img_dir_sort=natsorted(os.listdir(img_dir), key=lambda y:y.lower())

## convert all to greyscale, fix resolution and add to array
img_set = []
for img in img_dir_sort:
    img_array = cv2.imread(os.path.join(img_dir, img), cv2.IMREAD_GRAYSCALE)
    
    img_array = (cv2.resize(img_array,(60*res_scale,60*res_scale)))
    
    img_set.append(img_array)

## make img_set into numpy array and reshape
img_set = np.array(img_set).reshape(-1, 60*res_scale, 60*res_scale, 1)

# ======================================================================================
# LOAD LENS DATA FROM CREATED TXT

SB_txt = np.loadtxt(os.path.join(os.path.realpath('.'), 
                                 'dataset' + str(im_set) + '/SB' + str(im_set) + '.txt'
                                ))
SA_set = np.array((SB_txt)[:,2]) # power law values
SN_set = np.array((SB_txt)[:,1]) # substructure number

# ======================================================================================
# SAVE DATASETS WITH PICKLE

if im_set == 1:
    pickle_out = open('img_set.pickle', 'wb') #training images
    pickle.dump(img_set, pickle_out)
    pickle_out.close()

    pickle_out = open('SA_set.pickle', 'wb') #training set power laws
    pickle.dump(SA_set, pickle_out)
    pickle_out.close()

    pickle_out = open('SN_set.pickle', 'wb') #training set substructure numbers
    pickle.dump(SN_set, pickle_out)
    pickle_out.close()

if im_set == 2:
    pickle_out = open('test_set.pickle', 'wb') #test images
    pickle.dump(img_set, pickle_out)
    pickle_out.close()

    pickle_out = open('SA_test.pickle', 'wb') #test set power laws
    pickle.dump(SA_set, pickle_out)
    pickle_out.close()

    pickle_out = open('SN_test.pickle', 'wb') #test set substructure numbers
    pickle.dump(SN_set, pickle_out)
    pickle_out.close()