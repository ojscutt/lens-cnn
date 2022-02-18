# ======================================================================================
# IMPORT RELEVANT PACKAGES AND SET GRAPH PROPERTIES
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D
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

## normalise data
img_set = np.array(img_set/255.0)
SN_set = (SN_set - SN_set.min())/(SN_set.max()-SN_set.min())

# ======================================================================================
# TRAIN N-CNN

name = 'n_cnn_1' #enter n-cnn name here

start1 = time.time()

tensorboard = TensorBoard(log_dir=('logs/'+name))

n_cnn=Sequential()

## create convolutions
n_cnn.add(Conv2D(16, (5,5), padding='same'))
n_cnn.add(Activation('relu'))
n_cnn.add(BatchNormalization(axis=-1))
n_cnn.add(MaxPooling2D(pool_size=(2, 2)))
n_cnn.add(Dropout(0.05))

n_cnn.add(Conv2D(32, (5,5), padding='same'))
n_cnn.add(Activation('relu'))
n_cnn.add(BatchNormalization(axis=-1))
n_cnn.add(MaxPooling2D(pool_size=(2, 2)))
n_cnn.add(Dropout(0.05))

n_cnn.add(Conv2D(64, (5,5), padding='same'))
n_cnn.add(Activation('relu'))
n_cnn.add(BatchNormalization(axis=-1))
n_cnn.add(MaxPooling2D(pool_size=(2, 2)))
n_cnn.add(Dropout(0.05))

n_cnn.add(Flatten())
n_cnn.add(Dense(512))
n_cnn.add(Activation('relu'))
n_cnn.add(Dropout(0.05))

n_cnn.add(Dense(1, activation='linear'))

n_cnn.compile(loss='mean_squared_error',
            optimizer=Adam(learning_rate=0.0001))

n_cnn.fit(img_set, SN_set,
        validation_split=0.25,
        batch_size=16, epochs=30,
        callbacks=[tensorboard])

if not os.path.exists('models'):
    os.makedirs('models')

n_cnn.save('models/' + name + '.model')

end1 = time.time()
print(round(end1 - start1,2), 's')

# ======================================================================================
# TEST N-CNN

model = 'n_cnn_1' #enter name of model to test 

n_cnn = tf.keras.models.load_model((os.path.join(os.path.realpath('.'),
                                                 'models/'+model+'.model'
                                                )))

## load in relevant data
SN_set = np.array(pickle.load(open('SN_set.pickle', 'rb')))
test_set = pickle.load(open('test_set.pickle', 'rb'))
test_set = np.array(test_set/255.0)

SB_txt = np.loadtxt(os.path.join(os.path.realpath('.'), 'dataset2/SB2.txt'))
SN_test = np.array((SB_txt)[:,1])
SA_test = np.array((SB_txt)[:,2])

## test n-cnn and un-normalise predictions
SN_preds = n_cnn.predict([test_set])
SN_preds = (SN_preds*(SN_set.max()-SN_set.min()))+SN_set.min()

## store SN_preds for A-CNN
pickle_out = open('SN_preds.pickle', 'wb')
pickle.dump(SN_preds, pickle_out)
pickle_out.close()

# ======================================================================================
# PLOT + PRINT RESULTS

f, axes = plt.subplots(1, 1, figsize=(32,12))

## plot line of best fit for scatter
m, c = np.polyfit(SN_test, SN_preds, 1)
xaxes=(-1,2001)
plt.plot(xaxes, m*xaxes + c, 'r') 
plt.plot([-1000,1400], [-1000,1400],'k') #plot ideal line of best fit

## plot prediction scatter
plt.scatter(SN_test, SN_preds, marker='o',c=-SA_test, cmap='viridis')
cbar = plt.colorbar()
cbar.set_label('Power Law')

plt.ylabel('Predicted Number of Substructures')
plt.xlabel('Number of Substructures')
plt.legend(['Prediction Gradient','Ideal Gradient','Predictions'])
plt.ylim(0,1300)
plt.xlim(0,1300)


print(m)
p, V = np.polyfit(SN_test, SN_preds, 1, cov=True)
print("x_1: {} +/- {}".format(p[0], np.sqrt(V[0][0])))
print("x_2: {} +/- {}".format(p[1], np.sqrt(V[1][1])))

plt.title('Predicted Substructure Number Against True Value')