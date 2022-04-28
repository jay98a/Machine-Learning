'''
Name - Jay Samir Shah
'''

import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
import numpy as np
import time
import matplotlib.pyplot as plt

# ======================= Loading and preprocessing dataset ==============

# Dataset -> https://archive.ics.uci.edu/ml/datasets/Wine+Quality
data_read = np.genfromtxt("winequality-red.csv",delimiter=";",skip_header=1)
data = np.array(data_read)
print("1")

x = data[:,0:11]
x = preprocessing.normalize(x) # normalization of inputs
print(x)
y = data[:,11]
print(y)

# spliting dataset into training and testing samples
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)

# ================= One Hot Encoding for the output =====================

hot_en = preprocessing.OneHotEncoder(sparse=False)
f_y_train = hot_en.fit_transform(y_train.reshape(y_train.shape[0],1))
f_y_test = hot_en.fit_transform(y_test.reshape(y_test.shape[0],1))

# ================= Initialization of network =============================

n_in = x_train.shape[1]
n_hd = 10
n_ou = f_y_train.shape[1]

epoch = 10000
eta = 0.00001
momentum = 0.9
temp_acc = 0
err_tr = 0

# initialization of random weights
w1 = np.random.randn(n_hd,n_in)  # hidden layer 1
w2 = np.random.randn(n_hd,n_hd)  # hidden layer 2
w3 = np.random.randn(n_hd,n_hd)  # hidden layer 3
w4 = np.random.randn(n_hd,n_hd)  # hidden layer 4
out_w = np.random.randn(n_ou,n_hd)  # output layer

# ======================== Network Training ================================


# activation function
def sigmoid(x):
    return 1/(1 +np.exp(-x))


# derivative of activation function
def d_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))


rmse_arr = np.zeros(epoch)

# epoch loop
for epoch in range(0, epoch):

    cost_sum = 0
    err_tr = 0
    ee = np.zeros(x_test.shape[0])
    # data shuffle
    shuffle_seq = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_seq]
    f_y_train = f_y_train[shuffle_seq]
    # feed forward
    l1_out = sigmoid(x_train.dot(w1.T))
    l2_out = sigmoid(l1_out.dot(w2))
    l3_out = sigmoid(l2_out.dot(w3))
    l4_out = sigmoid(l3_out.dot(w4))
    l_out = sigmoid(l4_out.dot(out_w.T))

    rmse = np.sqrt(np.mean((l_out - f_y_train)**2))  # calculating rmse for this epoch
    rmse_arr[epoch] = rmse
    print("epoch {} rmse {}".format(epoch,rmse_arr[epoch]))

    # back propagate
    f_err = f_y_train - l_out
    out_delta = f_err * d_sigmoid(l4_out.dot(out_w.T))
    # out_delta = f_err * np.transpose(d_sigmoid(np.transpose(l4_out).dot(out_w)))

    err_l4 = out_delta.dot(out_w)
    l4_delta = err_l4 * d_sigmoid(l3_out.dot(w4))
    #l4_delta = err_l4 * np.transpose(d_sigmoid(l3_out.dot(w4.T)))

    err_l3 = l4_delta.dot(w4)
    l3_delta = err_l3 * d_sigmoid(l2_out.dot(w3))

    err_l2 = l3_delta.dot(w3)
    l2_delta = err_l2 * d_sigmoid(l1_out.dot(w2))

    err_l1 = l2_delta.dot(w2)
    l1_delta = err_l1 * d_sigmoid(x_train.dot(w1.T))

    # Updating weights
    w1 = w1 + (momentum * (eta * (np.transpose(l1_delta).dot(x_train))))
    w2 = w2 + (momentum * (eta * (np.transpose(l2_delta).dot(l1_out))))
    w3 = w3 + (momentum * (eta * (np.transpose(l3_delta).dot(l2_out))))
    w4 = w4 + (momentum * (eta * (np.transpose(l4_delta).dot(l3_out))))
    out_w = out_w + (momentum * (eta * (np.transpose(out_delta).dot(l4_out))))

end_time = time.process_time()

# loop for average Rmse
for k in range(len(rmse_arr)):
    temp_acc = temp_acc + rmse_arr[k]
a_rmse = (temp_acc/len(rmse_arr))
print("Average Training Rmse :",a_rmse)

# graph plot
plt.title("RMSE Curve")
plt.ylabel("MSE")
plt.xlabel("Epochs")
plt.plot(rmse_arr)
plt.show()

# ======================= Testing ===================================

err_te = 0
rmse_sum = 0
te_rmse_arr = np.zeros(x_test.shape[0])

for i in range(x_test.shape[0]):

    inp = x_test[i][:]
    l1_o = sigmoid(inp.dot(w1.T))
    l2_o = sigmoid(l1_o.dot(w2))
    l3_o = sigmoid(l2_o.dot(w3))
    l4_o = sigmoid(l3_o.dot(w4))
    l_o = sigmoid(l4_o.dot(out_w.T))
    y = np.argmax(l_o)
    d = np.argmax(f_y_test[i])
    te_rmse_arr[i] = np.sqrt(np.mean(y-d)**2)

    if d != y:
        err_te = err_te + 1

te_acc = err_te / x_test.shape[0]
print("Testing accuracy :",te_acc)

for j in (range(len(te_rmse_arr))):
    rmse_sum = rmse_sum + te_rmse_arr[j]

avg_rmse = rmse_sum/len(te_rmse_arr)
print("Average Testing Rmse :", avg_rmse)
