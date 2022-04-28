'''
Name - Jay Samir Shah
'''

import pandas as pd
import numpy as np
import sklearn.model_selection as sk
import matplotlib.pyplot as plt
import sklearn

# ************************** Read and prepare dataset *********************************
# https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
# https://archive.ics.uci.edu/ml/datasets/banknote+authentication

data_read = pd.read_csv("data_banknote_authentication.txt")
#data_read = pd.read_csv("transfusion.data")

data = np.array(data_read)
# data = pd.DataFrame(data_read)
# print(data)


x = data[:,0:4]
x = sklearn.preprocessing.normalize(x) # Normalize the input data
print(x)
y = data[:,4]
print(y)

# spliting dataset into training and testing samples
x_train,x_test,y_train,y_test = sk.train_test_split(x,y,test_size=0.20)

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
eta = 0.3
epoch = 600
err_array = [0] * epoch
temp_acc = 0
w = np.zeros(num_in)
#print(w)


# ********************** Training perceptron ************************

ee = np.zeros(x_train_join.shape[0])
mse = np.zeros(epoch)
m = np.zeros(num_in)
cost_store = [0] * epoch

shuffle_seq = np.random.permutation(np.arange(x_train_join.shape[0]))
data_shuffled = x_train_join[shuffle_seq,:]

y_pred_train = np.zeros(x_train.shape[0])

for i in range(epoch):
    shuffle_seq = np.random.permutation(range(x_train_join.shape[0]))
    data_shuffled_tr = data_shuffled[shuffle_seq]
    err_tr = 0
    cost_sum = 0
    m_sum = 0
    b_sum = 0

    # reducing learning rate with increasing epoch
    if i>=(epoch/4) and i<(epoch/3):
        eta = 0.1
    elif i>=(epoch/3) and i<(epoch/2):
        eta = 0.01
    elif i>=(epoch/2) and i<(epoch):
        eta = 0.00001

    for j in range(x_train_join.shape[0]):
        x = data_shuffled_tr[j][0:4] #inputs
        d = data_shuffled_tr[j][4] # desired output
        y_pred_train = np.sign((np.transpose(m).dot(x)) + b) # output
        if (y_pred_train == -1):
            y_pred_train = 0
        b = eta * (d-y_pred_train) # update bias
        ee[j] = d-y_pred_train # calculating error
        cost_sum = cost_sum + (ee[j] ** 2)
        m_sum = m_sum + (x * ee[j])
        b_sum = b_sum + ee[j]
        if (d - y_pred_train) != 0:
            err_tr = err_tr + 1 # err for accuracy
    #mse[i] = np.mean(ee**2)
    cost = ((1 / x_train_join.shape[0]) * cost_sum) # cost derivative
    m = m - (eta * ((-2 / x_train_join.shape[0]) * m_sum)) # m derivative
    # mse[i] = (2/train_samples)*m_sum
    cost_store[i] = cost
    b = b - (eta * (-2 / x_train_join.shape[0]) * b_sum) # bias derivative
    # print("m {},b {},c {}, epoch {}".format(m, b, cost, i))
    err_array[i] = 100 - ((err_tr/x_train_join.shape[0])*100) # error array store
# ***************** for calculating average accuracy ************
for k in range(len(err_array)):
    temp_acc = temp_acc + err_array[k]
acc = (temp_acc/len(err_array))
print("Average Training accuracy :",acc)

plt.plot(cost_store)  # mse plots
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
    y_pred = int(np.sign(np.transpose(m).dot(x) + b)) #output
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


