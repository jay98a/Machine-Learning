'''
Name - Jay Samir Shah
Student Id -1146105
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D as convolution_layer
import tensorflow.keras.layers as layer_operation
import matplotlib.pyplot as plt

#======================= Transfer learing (loading weights) ==================================
#============================= Loading and Preproccessing Dataset ==============================
(x, y), (x1, y1) = keras.datasets.cifar10.load_data()

# Normalizing data
#x_train = np.mean(x)
x_train = x / 255
x_test = x1 / 255

# One hot encoding
y_train = keras.utils.to_categorical(y)
y_test = keras.utils.to_categorical(y1)

def reshape_images(image, label):
  image = tf.image.resize(image, (227, 227))
  return image, label

train_join = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_join = (train_join.map(reshape_images).batch(batch_size=32, drop_remainder=True)) # batch size 32 = 1562 in a batch

test_join = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_join = (test_join.map(reshape_images).batch(batch_size=32, drop_remainder=True))


#=============================== Model Initialization (Alexnet) =========================

model = keras.models.Sequential([
  # convolution layer 1
  convolution_layer(96, (11,11),strides=(4,4), activation='relu', input_shape=(227, 227, 3)),
  layer_operation.BatchNormalization(),
  layer_operation.MaxPooling2D(2, strides=(2,2)),
  # convolution layer 2
  convolution_layer(256, (11,11),strides=(1,1), activation='relu',padding="same"),
  layer_operation.BatchNormalization(),
  # convolution layer 3
  convolution_layer(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  layer_operation.BatchNormalization(),
  # convolution layer 4
  convolution_layer(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  layer_operation.BatchNormalization(),
  # convolution layer 5
  convolution_layer(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
  layer_operation.BatchNormalization(),
  layer_operation.MaxPooling2D(2, strides=(2, 2)),
  # layer flatteining
  layer_operation.Flatten(),
  # fully connected layer 1
  layer_operation.Dense(4096, activation='relu'),
  # fully connected layer 2
  layer_operation.Dense(4096, activation='relu'),
  layer_operation.Dense(10, activation='softmax')
])

# Loading weights for network
# the following weights can also be downloaded from https://drive.google.com/file/d/1dv86jnibgzZJ2QuoYm4t5GzRsRbZ6Vsf/view?usp=sharing
model.load_weights('cifar10_weights.h5')

optimizer = keras.optimizers.SGD(learning_rate=0.0008, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

# ===========================Training and Testing the model =================================
trained_model = model.fit(train_join, epochs=20, validation_data=(test_join))

#====================== Ploting the results =====================================

plt.plot(trained_model.history['accuracy'])
plt.plot(trained_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()