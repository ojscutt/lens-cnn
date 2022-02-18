# ======================================================================================
# IMPORT RELEVANT PACKAGES AND SET GRAPH PROPERTIES
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Concatenate
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import cv2
import os
import seaborn as sns
sns.set_theme(font="Cambria", style='whitegrid',font_scale=2.5)
plt.rcParams["axes.labelsize"] = 50

# ======================================================================================
# LOAD IN PICKLED DATASETS AND NORMALISE

img_set = pickle.load(open('img_set.pickle', 'rb'))
SN_set = pickle.load(open('SN_set.pickle', 'rb'))
SA_set = pickle.load(open('SA_set.pickle', 'rb'))

## normalise data
img_set = np.array(img_set/255.0)
SN_set = (SN_set - SN_set.min())/(SN_set.max()-SN_set.min())
SA_set = (SA_set - SA_set.min())/(SA_set.max()-SA_set.min())

# ======================================================================================
# TRAIN A-CNN

name = 'a_cnn_1' #enter a-cnn name here
res_scale = 1 #set desired resolution scaling factor (60x60=1, 30x30=0.5 etc.)

start1 = time.time()

tensorboard = TensorBoard(log_dir=('logs/'+name))

## define inputs
img_input = Input((60*res_scale,60*res_scale,1))
num_input = Input((1,))

## define convolutions
img_conv = Conv2D(8, (3,3), padding='same')(img_input)
img_conv = Activation('relu')(img_conv)
img_conv = BatchNormalization(axis=-1)(img_conv)
img_conv = MaxPooling2D(pool_size=(2, 2))(img_conv)
img_conv = Dropout(0.25)(img_conv)

img_conv = Flatten()(img_conv)
img_conv = Dense(512)(img_conv)
img_conv = Activation('relu')(img_conv)
img_conv = Dropout(0.25)(img_conv)

## concatenate layer
concat_layer = Concatenate()([num_input, img_conv])
output = (Dense(1, activation='linear'))(concat_layer)

a_cnn = Model(inputs=[img_input, num_input], outputs=output)

a_cnn.compile(loss='mean_absolute_error',
             optimizer=Adam(learning_rate=0.0001))

a_cnn.fit([img_set, SN_set], SA_set,
          validation_split=0.25,
          batch_size=16, epochs=30,
          callbacks=[tensorboard])

if not os.path.exists('models'):
    os.makedirs('models')

a_cnn.save('models/' + name + '.model')

end1 = time.time()
print(round(end1 - start1,2), 's')

# ======================================================================================
# TEST A-CNN

model='a_cnn_1' #enter name of model to test 

a_cnn = tf.keras.models.load_model((os.path.join(os.path.realpath('.'),
                                                 'models/'+model+'.model'
                                                )))

## load in relevant data
SA_set = np.array(pickle.load(open('SA_set.pickle', 'rb')))
SN_set = np.array(pickle.load(open('SN_set.pickle', 'rb')))
test_set = pickle.load(open('test_set.pickle', 'rb'))
test_set = np.array(test_set/255.0)

SB_txt = np.loadtxt(os.path.join(os.path.realpath('.'), 'dataset2/SB2.txt'))
SN_test = np.array((SB_txt)[:,1])
SN_test = (SN_test - SN_set.min())/(SN_set.max()-SN_set.min())
SA_test = np.array((SB_txt)[:,2])

SN_preds = np.array(pickle.load(open('SN_preds.pickle', 'rb')))
SN_preds = (SN_preds - SN_set.min())/(SN_set.max()-SN_set.min())

## test a-cnn and un-normalise predictions
SA_preds = a_cnn.predict([test_set,SN_test])
SA_preds = ((SA_preds*(SA_set.max()-SA_set.min()))+SA_set.min())

# ======================================================================================
# PLOT + PRINT RESULTS

f, axes = plt.subplots(1, 1, figsize=(32,12))

## plot line of best fit for scatter
m, c = np.polyfit(SA_test, SA_preds, 1)
plt.plot(SA_test, m*SA_test + c, 'r')
plt.plot(SA_test, SA_test,'k') #plot ideal line of best fit

## plot prediction scatter
plt.scatter(SA_test, SA_preds, marker='o',c=SN_test, cmap='viridis') #plot actual/predicted values
cbar = plt.colorbar()
cbar.set_label('Number of substructures')

plt.ylabel('Predicted Substructure Power Law')
plt.xlabel('Substructure Power Law')
plt.legend(['Prediction Gradient','Ideal Gradient','Predictions'],bbox_to_anchor=(1.4,1))

print(m)
p, V = np.polyfit(SA_test, SA_preds, 1, cov=True)
print("x_1: {} +/- {}".format(p[0], np.sqrt(V[0][0])))
print("x_2: {} +/- {}".format(p[1], np.sqrt(V[1][1])))

plt.title('Substructure Power Law Against Actual Value')