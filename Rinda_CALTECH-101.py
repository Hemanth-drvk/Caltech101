#!/usr/bin/env python
# coding: utf-8

# # Caltech-101 Image Classification

# ### Importing the libraries

# In[18]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers
import tensorflow as tf
from keras.callbacks import EarlyStopping
import os
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

from warnings import simplefilter
simplefilter(action='ignore', category = DeprecationWarning)


# ### Loading the images

# In[20]:


def load_dataset(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.transpose(np.array([img, img, img]), (2, 0, 1))
    return img
    
path = "C:/Users/krish/Desktop/COMP5013/101_ObjectCategories"
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print ("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

folders = sorted(os.listdir(path))
nfolders = len(folders)
imgs = []
labels = []


# ### Scaling and normalizing pictures

# In[21]:


for i, folder in enumerate(folders):
    iter = 0
    for f in os.listdir(path + "/" + folder):
        if iter == 0:
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_exts:
                continue
            fullpath = os.path.join(path + "/" + folder, f)
            img = scipy.misc.imresize(load_dataset(fullpath), [128,128,3])
            img = img.astype('float32')
            img[:,:,0] -= 123.68
            img[:,:,1] -= 116.78
            img[:,:,2] -= 103.94
            imgs.append(img) 
            label_curr = i
            labels.append(label_curr)
        
print ("Num imgs: %d" % (len(imgs)))
print ("Num labels: %d" % (len(labels)))
print (nfolders)


# ### Splitting the data

# In[4]:


#splitting the data into train test and split
X_train, X_test, y_train, y_test = train_test_split(imgs, labels,test_size = 0.30)
X_train = np.stack(X_train, axis=0)
y_train = np.stack(y_train, axis=0)
X_test = np.stack(X_test, axis=0)
y_test = np.stack(y_test, axis=0)
print ("Num train_imgs: %d" % (len(X_train)))
print ("Num test_imgs: %d" % (len(X_test)))

# one hot encode target values
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes= y_test.shape[1]


# ### Initialising the default values

# In[5]:


seed = 7
np.random.seed(seed)

epochs =200
lrate = 0.01
decay = lrate/epochs

np.random.seed(seed)


# ### Defining and running Baseline : 4 VGG Block for 100 epochs 

# In[10]:


model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(128,128,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
    
opt = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
model.summary()

history = model.fit(X_train, y_train,batch_size=64,epochs = 100,validation_data=(X_test, y_test), shuffle=True)

model.save('model_appraoch_1_100.h5')
#evaluating the test dataset
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

pyplot.subplot(211)
pyplot.ylabel('Cross Entropy Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','test'])
plt.title('loss')
#plt.savefig("loss7.png",dpi=300,format="png")
plt.figure()
pyplot.subplot(212)
pyplot.ylabel('Classification Accuracy')
plt.plot(history.history['acc'], color = 'blue', label = 'train')
plt.plot(history.history['val_acc'], color = 'orange', label = 'test')
plt.legend(['train','test'])
plt.title('accuracy')
pyplot.show()
pyplot.close()


# ### Defining and running Baseline : 8 VGG Block running for 200 epochs with Dropout Regularization

# In[6]:


model = Sequential()
    
model.add(Conv2D(64, (3, 3), input_shape=(128,128,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
    
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
    
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
    
opt = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
print(model.summary())


history = model.fit(X_train, y_train, batch_size=64, validation_data=(X_test, y_test),epochs=200, shuffle=True)

# save model
model.save('model_approach_2_200.h5')
#evaluating the test dataset
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

pyplot.subplot(211)
pyplot.ylabel('Cross Entropy Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','test'])
plt.title('loss')
#plt.savefig("loss7.png",dpi=300,format="png")
plt.figure()
pyplot.subplot(212)
pyplot.ylabel('Classification Accuracy')
plt.plot(history.history['acc'], color = 'blue', label = 'train')
plt.plot(history.history['val_acc'], color = 'orange', label = 'test')
plt.legend(['train','test'])
plt.title('accuracy')
pyplot.show()
pyplot.close()


# ### Defining and running Baseline : 8 VGG Block for 400 epochs with Dropout Regularization and Data Augmentation

# In[7]:


model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

decay = 0.01/400
sgd = SGD(lr=0.01, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

np.random.seed(seed)

datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(X_train)
hist = model.fit(datagen.flow(X_train, y_train,  batch_size=64), validation_data=(X_test, y_test),epochs=400, shuffle=True)


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['train','test'])
plt.title('loss')
plt.savefig("loss7.png",dpi=300,format="png")
plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['train','test'])
plt.title('accuracy')
plt.savefig("accuracy7.png",dpi=300,format="png")


# In[8]:


model.save('model_appraoch3_400.h5')


# ### Evaluating the Approach-1  on test set

# #### Loading the model_appraoch_1_100.h5

# In[22]:


from keras.models import load_model

model_test = load_model('model_appraoch_1_100.h5')
scores = model_test.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# ### Evaluating the Approach-2  on test set

# #### Loading the model_approach_2_200.h5

# In[23]:


from keras.models import load_model

model_test = load_model('model_approach_2_200.h5')
scores = model_test.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# ### Evaluating the Approach-3  on test set

# #### Loading the model_approach3_400.h5

# In[16]:


from keras.models import load_model

model_test = load_model('model_appraoch3_400.h5')
scores = model_test.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# Rinda Digamarthi(157742d)

# In[ ]:




