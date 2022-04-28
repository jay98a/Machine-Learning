'''
Name - Jay Samir Shah
'''

import pandas as pd
import numpy as np
import sklearn.model_selection as sk
import matplotlib.pyplot as plt

# ************************** Read and prepare dataset *********************************
# https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
data_read = pd.read_csv("transfusion.data")

data = np.array(data_read)
# data = pd.DataFrame(data_read)
# print(data)

x = data[:,0:4]
print(x)
y = data[:,4]
print(y)

# spliting dataset into training and testing samples
x_train,x_test,y_train,y_test = sk.train_test_split(x,y,test_size=0.2)

# transposing y_train for joining it again with data for shuffle
temp_y = np.array(y_train)
yt = np.array([temp_y])
trans = yt.transpose()

#x_train_join = np.concatenate[(x_train,trans)]
x_train_join = np.append(x_train,trans, axis=1)


# ************************* Initialization of perceptron *********************

num_in = 4
b = 0
err_tr = 0
err_te = 0
eta = 0.95
epoch = 800
err_array = [0] * epoch
temp_acc = 0
w = np.zeros(num_in)
#print(w)


# ********************** Training perceptron ************************

ee = np.zeros(x_train_join.shape[0])
mse = np.zeros(epoch)


shuffle_seq = np.random.permutation(np.arange(x_train_join.shape[0]))
data_shuffled = x_train_join[shuffle_seq,:]

y_pred_train = np.zeros(x_train.shape[0])

for i in range(epoch):
    shuffle_seq = np.random.permutation(range(x_train_join.shape[0]))
    data_shuffled_tr = data_shuffled[shuffle_seq]
    err_tr = 0

    if i>=(epoch/4) and i<(epoch/3):
        eta = 0.7
    elif i>=(epoch/3) and i<(epoch/2):
        eta = 0.45
    elif i>=(epoch/2) and i<epoch:
        eta = 0.1

    for j in range(x_train_join.shape[0]):
        x = data_shuffled_tr[j][0:4] #inputs
        d = data_shuffled_tr[j][4] # desired output
        y_pred_train = np.sign((np.transpose(w).dot(x)) + b) # output
        if (y_pred_train == -1):
            y_pred_train = 0
        b = eta * (d-y_pred_train) # update bias
        ee[j] = d-y_pred_train # calculating error
        w_new = w + (eta*(ee[j])*x) # update weight
        w = w_new
        if (d - y_pred_train) != 0:
            err_tr = err_tr + 1 # err for accuracy
    print(i," Epoch Done")
    mse[i] = np.mean(ee**2)
    err_array[i] = 100 - ((err_tr/x_train_join.shape[0])*100)
# ***************** for calculating average accuracy ************
for k in range(len(err_array)):
    temp_acc = temp_acc + err_array[k]
acc = (temp_acc/len(err_array))
print("Average Training accuracy :",acc)

plt.plot(mse) # mse plots
plt.show()
plt.plot()

# ************************** Testing *************************

temp1 = []
temp2 = []
y_pos = 0
y_neg = 0

y_pred = np.zeros(x_test.shape[0])
for i in range(x_test.shape[0]):

    x = x_test[i][:] #input
    d = y_test[i] #desired output
    y_pred = int(np.sign(np.transpose(w).dot(x) + b)) #output
    if(y_pred==-1):
        y_pred = 0
    temp1.append(d)
    temp2.append(y_pred)
    if y_pred == 1: # y addition and substraction logic for ploting
        y_pos = y_pos +1
    if y_pred == 0:
        y_neg= y_neg+1
    if (d - y_pred) != 0:
        err_te = err_te + 1
        if y_pred == 1:
            y_pos = y_pos -1
        else:
            y_neg = y_neg -1

print(temp1)
print(temp2)
#print(err)
#print(err/x_test.shape[0])

print("Testing Accuracy: ",(100-(err_te/x_test.shape[0])*100))
print("-----------------", y_pos)
print("-----------------",y_neg)

# ****************************** plotting the graph ************************
height = [y_pos, y_neg]
bars = ('Not Donated', 'Donated')
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.show()



