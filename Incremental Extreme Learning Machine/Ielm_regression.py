'''
# Name - Jay Samir Shah
'''

import numpy as np
import pandas as pd
import time
import math
from sklearn.model_selection import train_test_split as split
import sklearn as sk

#===========================reading dataset ====================================
# Dataset link --> https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant

dataSet = pd.read_csv('Folds5x2_pp.csv')
data = np.array(dataSet)

# ========================== Normalize and Test Train Split Data =========================

x = data[:,0:3]
#print(x)
x = sk.preprocessing.normalize(x) # Normalizing the input data
y = data[:,3]
x_train, x_test, y_train, y_test = split(x,y,train_size=0.8)  # splitting the data

#=====================================Initializing Variables ======================

hiddenNeurons = 200 # number of hidden neurons
err_int = 0.9 # starting error

#===================================== Training =========================================
tr_start_time = time.time()

for i in range(hiddenNeurons):
    H = np.random.normal(size=[x_train.shape[0], i + 1])
    Error = y_train

    for j in range(x_train.shape[0]):
        x = np.transpose(x_train[j]) # taking the input values (x)
        b = np.random.random(size=[i+1]) # random bias
        pos = 0
        h_temp = []

        for k in range(i+1):
            a = np.random.random((x_train.shape[1], 1))
            mul = math.sin(np.dot(x, a) + b ) # (x.a+b)
            h_temp.insert(k, mul)

        h_temp = np.reshape(h_temp, (-1, i+1)) # temp h array store for inner loop
        H[pos] = h_temp[k] # final H store
        pos = pos + 1

    beta = np.dot(np.linalg.pinv(H), mul) # taking H inverse for mul

    cal_error = Error - np.dot(H, beta)  # Calculating Actual Error
    # computing the RMSE

    rmse = (np.sqrt((y_train - cal_error)**2).mean()) # calculating rmse
    if (rmse < err_int):  # loop break condition if rmse increases more than initial error
        print("Training RMSE : ", rmse)
        break

tr_end_time = time.time()
print("Training Time Taken : ",abs(tr_start_time-tr_end_time))

#================================= Testing ==========================================

start_time = time.time()

H = np.zeros(shape=(x_test.shape[0], beta.shape[0]))

for i in range(x_test.shape[0]): #for all the test data

    x = np.transpose(x_test[i])  # taking the input values (x)
    pos = 0
    h_temp = []
    for j in range(beta.shape[0]):

        pred_y = math.sin(np.dot(x, a) + b) # predicted y
        h_temp.insert(j, pred_y)

    h_temp = np.reshape(h_temp, (-1, beta.shape[0]))
    H[pos] = h_temp[0]
    pos = pos + 1
err = 1 / (1 + np.exp(np.dot(H, beta))) #err

d = 1 / (1 + np.exp(y_test))   #sigmoid of the desired output
d = np.array(d)
d = np.expand_dims(d, axis=-1)
rmse = (np.sqrt((d - err)**2).mean()) # calculating rmse

end_time = time.time()

print("Testing RMSE", rmse)
print("Testing Time Taken : ",abs(start_time-end_time))
