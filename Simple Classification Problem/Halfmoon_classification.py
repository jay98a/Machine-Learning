'''
Name -  Jay Samir Shah
'''

import numpy as np
import math
import matplotlib.pyplot as plt
import time

def halfmoon_fun(rad,width,d,n):

    data = np.zeros((3,n))

    if rad < (width / 2):
        print('The radius should be at least larger than half the width')
        return 1

    if n % 2 != 0:
        print('Please make sure the number of samples is even')
        return 1

    aa = np.random.random_sample((2, int(n/2)))
    radius = (rad - width/2) + width * aa[0]
    theta = math.pi * aa[1]

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    label = 1 * np.ones((1, len(x)))  # label for Class 1

    x1 = radius * np.cos(-theta) + rad
    y1 = radius * np.sin(-theta) - d
    label1 = -1 * np.ones((1, len(x1)))   # label for Class 2


    data[0,:] = np.append(x,x1)
    data[1,:] = np.append(y,y1)
    data[2,:] = np.append(label,label1)

    shuffle_seq = np.random.permutation(np.arange(n))
    data_shuffled = np.transpose(data[:, shuffle_seq])

    return [data,data_shuffled]

# *************************** Variables for halfmoon *************************

rad = 10
width = 6
d = 0
train_samples = 1000
test_samples = 4000
n = train_samples + test_samples

[data,data_shuffled] = halfmoon_fun(rad,width,d,n)
print(data)

# *************** Initialization of perceptron *******************

num_in = 2  # number of input neuron
b = 0
err = 0
eta = 0.95
epoch = 50
err_tr = 0
err_array = [0] * epoch
temp_acc = 0
#w = np.array([0.,0.,0.])
#w = [[0.],[0.],[0.]]
#w = np.zeros(num_in)
w = np.empty(num_in)
print(w)

# ********************** Training perceptron ************************

st = time.process_time()
ee = np.zeros(train_samples)
mse = np.zeros(epoch)

for i in range(epoch):
    shuffle_seq = np.random.permutation(range(train_samples))
    data_shuffled_tr = data_shuffled[shuffle_seq]
    err_tr = 0
    for j in range(train_samples):
        x = data_shuffled_tr[j][0:2]  # inputs
        # print(x)
        #x = np.insert(x,0,1)
        # print("after :",x)
        d = int(data_shuffled_tr[j][2]) # desired output
        y = int(np.sign(np.transpose(w).dot(x)) + b)
        #y= np.sign(w*x)
        b = eta * (d-y)
        ee[j] = d-y
        w_new = w +eta*(ee[j])*x
        w = w_new
        if d != y:
            err_tr = err_tr + 1 # err for accuracy

    mse[i] = np.mean(ee**2)
    err_array[i] = 100 - ((err_tr/train_samples)*100)

# ***************** for calculating average accuracy ************
for k in range(len(err_array)):
    temp_acc = temp_acc + err_array[k]
#acc = (temp_acc/epoch)
acc = err_array[-1]
print("Average Training accuracy :",acc)

plt.plot(mse)
plt.show()
plt.plot()

# ******************************** Testing **************************

shuffle_seq = np.random.permutation(range(test_samples))
data_shuffled_te = data_shuffled[shuffle_seq]

for i in range(test_samples):
    x = data_shuffled_te[i][0:2]
    d = int(data_shuffled_te[i][2])
    y = int(np.sign(np.transpose(w).dot(x)))
    if y == 1:
        plt.plot(x[0], x[1], 'rx')
    if y == -1:
        plt.plot(x[0], x[1], 'gx')
    if (d - y) != 0:
        err = err + 1

print("Testing Accuracy : ",100-(err/test_samples)*100)

plt.title("Classification using Perceptron")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()