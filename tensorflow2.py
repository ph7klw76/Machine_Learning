
"""
Created on Sat Nov  9 13:34:04 2019
Journal of Molecular Graphics and Modelling
Volume 105, June 2021, 107891
@author: user
"""

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
datatosave=[]
matrices = np.load('E:/newdescriptor.npy')
molden = np.load('E:/homolumost2.npy')
n_samples = len(matrices)
X = matrices
y = ((molden*-1)/27.2114)
X = StandardScaler().fit_transform(X)


i=0   #........#

# def scheduler(epoch):
#     if epoch < 100000:
#         return 0.001*(i+1)
#     else:
#         return 0.001*(i+1)*np.exp(0.001 * (1000 - epoch))
xx=5000   
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(378,)),
    tf.keras.layers.Dense(80, activation='tanh',kernel_regularizer=keras.regularizers.l2(0.0001)),
    tf.keras.layers.AlphaDropout(rate=0.0005), 
    tf.keras.layers.Dense(20, activation='tanh',kernel_regularizer=keras.regularizers.l2(0.0001)),
    tf.keras.layers.AlphaDropout(rate=0.0005), 
    tf.keras.layers.Dense(4, activation='tanh')])

testsize=['0.2']
learningrate=['0.0005']
for testsize_1 in testsize:
    testsize_1=float(testsize_1)
    for learningrate_1 in learningrate:
        learningrate_1=float(learningrate_1)
        for i in range(1):
            def scheduler(epoch):
                if epoch < 1000:
                    return learningrate_1*(i+1)
                else:
                    return learningrate_1*(i+1)*np.exp(0.001 * (1000 - epoch))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningrate_1*(i+1)),
                            loss=tf.keras.losses.MeanAbsoluteError(),
                            metrics=['MeanAbsoluteError'])
            X_train1, X_test, y_train1, y_test = train_test_split(X, y,test_size = testsize_1, random_state=1)
            X_test2, X_vali, y_test2, y_vali = train_test_split(X_test,y_test,test_size = 0.5, random_state=1)
            print(testsize_1)
            callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
            history=model.fit(X_train1,y_train1, batch_size=400, validation_data=(X_vali, y_vali), epochs=xx, shuffle= True, callbacks=[callback])
            prediction=model.predict(X_test)*27.2114*-1
            prediction2=model.predict(X_train1)*27.2114*-1   #........#
            a=history.history['mean_absolute_error'][xx-1]
            b=history.history['val_mean_absolute_error'][xx-1]
            R2 = r2_score(y_test*27.2114*-1, prediction)
            datatosave.append([learningrate_1*(i+1),a,b,R2, testsize_1]) 
            print(R2)

np.savetxt('E:/testNN20expo-different fraction.txt',datatosave)
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.yscale('log')
plt.title('Model MeanAbsoluteError')
plt.ylabel('MeanAbsoluteError')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('MeanAbsoluteError.png',dpi=600)
plt.show()

model.save_weights('E:/weight.h5') #........#

a = plt.axes(aspect='equal')
plt.scatter(y_test*27.2114*-1, prediction,s=0.015, c='red')
plt.scatter(y_train1*27.2114*-1, prediction2,s=0.015, c='green')
plt.xlabel('Energy Levels calculated using DFT [eV]')
plt.ylabel('Energy Levels predicted using DNN [eV]')
lims = [-8.5, 6.0]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig('E:/prediction.png', dpi=600)

