import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
#%%
iris =load_iris()
X = iris.data[:,(2,3)]
y = (iris.target==0).astype(np.int)
per_clf = Perceptron()
per_clf.fit(X,y)
y_pred=per_clf.predict([[2,0.5]])
#%%
import tensorflow
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full,y_train_full),(X_test,y_test) = fashion_mnist.load_data()
#%%
print(X_train_full.shape)
print(y_train_full.shape)
#%%%
X_valid,X_train = X_train_full[:5000] / 255.0,X_train_full[5000:] / 255.0
y_valid,y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test/ 255.0
#%%
class_names = ["T-shirt/top","Trousers","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag",
               "Ankle boot"]
print(class_names[y_train[0]])
#%%
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation="relu",kernel_initializer="he_normal"))
model.add(keras.layers.Dense(100,activation="relu",kernel_initializer="he_normal"))
model.add(keras.layers.Dense(10,activation="softmax"))
#%%
model.summary()
#%%
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
history = model.fit(X_train,y_train,epochs=30,validation_data=(X_valid,y_valid))
#%%
import pandas as pd 
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
#%%
model.evaluate(X_test,y_test)
#%% using the model to make predictions
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))
#%%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full,X_test,y_train_full,y_test = train_test_split(housing.data,housing.target)
X_train,X_valid,y_train,y_valid = train_test_split(X_train_full,y_train_full)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
#%%
model = keras . models . Sequential ([ keras . layers . Dense ( 30 , activation = "relu" , input_shape = X_train . shape [ 1 :]), keras . layers . Dense ( 1 ) ]) 
model . compile ( loss = "mean_squared_error" , optimizer = "sgd" )
history = model . fit ( X_train , y_train , epochs = 20 , validation_data = ( X_valid , y_valid )) 
mse_test = model . evaluate ( X_test , y_test ) 
X_new = X_test [: 3 ] # pretend these are new instances 
y_pred = model . predict ( X_new )
#%% Batch normalization
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(300,activation="selu",kernel_initializer="lecun_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(200,activation="selu",kernel_initializer="lecun_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(100,activation="selu",kernel_initializer="lecun_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(10,activation="softmax"))
#%%
model.summary()
#%%
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
history = model.fit(X_train,y_train,epochs=30,validation_data=(X_valid,y_valid))
#%%
import pandas as pd 
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
#%%
model.evaluate(X_test,y_test)
#%% Regulization for aviding overfitting
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(300,activation="selu",kernel_initializer="lecun_normal",
                             kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(200,activation="selu",kernel_initializer="lecun_normal",
                             kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(100,activation="selu",kernel_initializer="lecun_normal",
                             kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(10,activation="softmax"))
#%%
model.summary()

#%%
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
history = model.fit(X_train,y_train,epochs=200,validation_data=(X_valid,y_valid))
#%%
import pandas as pd 
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
#%%
model.evaluate(X_test,y_test)
#%% Computational Graph
import tensorflow as tf
x1 = tf.constant(1)
x2=tf.constant(2)
z = tf.add(x1,x2)
sess = tf.Session()
print(sess.run(z))

#%%
'''
  Keras model discussing Categorical (multiclass) Hinge loss.
'''
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from mlxtend.plotting import plot_decision_regions

# Configuration options
num_samples_total = 3000
training_split = 1000
num_classes = 3
feature_vector_length = len(range(0,1000))
input_shape = (1000,2)
loss_function_used = 'categorical_hinge'

additional_metrics = ['accuracy']
num_epochs = 30
batch_size = 5
validation_split = 0.2 # 20%

# Generate data
X, targets = make_blobs(n_samples = num_samples_total, centers = [(0,0), (15,15), (0,15)], n_features = num_classes, center_box=(0, 1), cluster_std = 1.5)
categorical_targets = to_categorical(targets)
X_training = X[training_split:, :]
X_testing = X[:training_split, :]
Targets_training = categorical_targets[training_split:]
Targets_testing = categorical_targets[:training_split].astype(np.integer)

# Generate scatter plot for training data
plt.scatter(X_training[:,0], X_training[:,1])
plt.title('Three clusters ')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Create the model
model = Sequential()
model.add(Dense(4, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(2, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(num_classes, activation='tanh'))
# Configure the model and start training
model.compile(loss=loss_function_used, optimizer="adam", metrics=additional_metrics)
history = model.fit(X_training, Targets_training, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=validation_split)

# Test the model after training
test_results = model.evaluate(X_testing, Targets_testing, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

'''
  The Onehot2Int class is used to adapt the model so that it generates non-categorical data.
  This is required by the `plot_decision_regions` function.
  The code is courtesy of dr. Sebastian Raschka at https://github.com/rasbt/mlxtend/issues/607.
  Copyright (c) 2014-2016, Sebastian Raschka. All rights reserved. Mlxtend is licensed as https://github.com/rasbt/mlxtend/blob/master/LICENSE-BSD3.txt.
  Thanks!
'''
# No hot encoding version
class Onehot2Int(object):

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        y_pred = self.model.predict(X)
        return np.argmax(y_pred, axis=1)

# fit keras_model
keras_model_no_ohe = Onehot2Int(model)

# Plot decision boundary
plot_decision_regions(X_testing, np.argmax(Targets_testing, axis=1), clf=keras_model_no_ohe, legend=3)
plt.show()
'''
  Finish plotting the decision boundary.
'''
#
# Visualize training process
plt.plot(history.history['loss'], label='Categorical Hinge loss (training data)')
plt.plot(history.history['val_loss'], label='Categorical Hinge loss (validation data)')
plt.title('Categorical Hinge loss for circles')
plt.ylabel('Categorical Hinge loss value')
plt.yscale('log')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

#%% learning rate finder
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from keras_lr_finder import LRFinder
#%%
# Model configuration
batch_size = 250
img_width, img_height, img_num_channels = 28, 28, 1
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 5
start_lr = 0.0001
end_lr = 1
moving_average = 20
# Determine tests you want to perform
tests = [
  (SGD(), 'SGD optimizer'),
  (Adam(), 'Adam optimizer'),
]
# Set containers for tests
test_learning_rates = []
test_losses = []
test_loss_changes = []
labels = []
# Perform each test
for test_optimizer, label in tests:

  # Compile the model
  model.compile(loss=loss_function,
                optimizer=test_optimizer,
                metrics=['accuracy'])

  # Instantiate the Learning Rate Range Test / LR Finder
  lr_finder = LRFinder(model)

  # Perform the Learning Rate Range Test
  outputs = lr_finder.find(X_train, y_train, start_lr=start_lr, end_lr=end_lr, batch_size=batch_size, epochs=no_epochs)

  # Get values
  learning_rates  = lr_finder.lrs
  losses          = lr_finder.losses
  loss_changes = []

  # Compute smoothed loss changes
  # Inspired by Keras LR Finder: https://github.com/surmenok/keras_lr_finder/blob/master/keras_lr_finder/lr_finder.py
  for i in range(moving_average, len(learning_rates)):
    loss_changes.append((losses[i] - losses[i - moving_average]) / moving_average)

  # Append values to container
  test_learning_rates.append(learning_rates)
  test_losses.append(losses)
  test_loss_changes.append(loss_changes)
  labels.append(label)

#%%
# Generate plot for Loss Deltas
for i in range(0, len(test_learning_rates)):
  plt.plot(test_learning_rates[i][moving_average:], test_loss_changes[i], label=labels[i])
plt.xscale('log')
plt.legend(loc='upper left')
plt.ylabel('loss delta')
plt.xlabel('learning rate (log scale)')
plt.title('Results for Learning Rate Range Test / Loss Deltas for Learning Rate')
plt.show()

# Generate plot for Loss Values
for i in range(0, len(test_learning_rates)):
  plt.plot(test_learning_rates[i], test_losses[i], label=labels[i])
plt.xscale('log')
plt.legend(loc='upper left')
plt.ylabel('loss')
plt.xlabel('learning rate (log scale)')
plt.title('Results for Learning Rate Range Test / Loss Values for Learning Rate')
plt.show()


