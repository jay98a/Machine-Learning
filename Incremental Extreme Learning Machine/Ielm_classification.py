'''
# Name - Jay Samir Shah
'''

import tensorflow as t
import sklearn.preprocessing as sk
import time
import numpy as np
import math
import requests
requests.packages.urllib3.disable_warnings()
# please import SSL certificate if getting an error while running

(x_train, y_train), (x_test, y_test) = t.keras.datasets.mnist.load_data()

#=====================================Initializing Variables ======================

hiddenNeurons = 20 # number of hidden neurons
err_int = 0.9 # starting error

#===================================== Training =========================================
tr_start_time = time.time()

for i in range(hiddenNeurons):
    H = np.random.normal(size=[x_train.shape[0], i + 1])
    Error = y_train

    for j in x_train:
        #x = np.transpose(x_train[j]) # taking the input values (x)
        x = np.reshape(j, (1, np.product(j.shape)))
        b = np.random.random(size=[i+1]) # random bias
        pos = 0
        h_temp = []

        for k in range(i+1):
            a = np.random.random((x.shape[1], 1))
            #a = np.reshape(a, (-1, a.shape[1]))
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

