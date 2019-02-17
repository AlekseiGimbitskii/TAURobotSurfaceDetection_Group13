import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.datasets import imdb

from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from matplotlib import pyplot
from keras.layers import ConvLSTM2D


# fit and evaluate a model
def evaluate_model(X_train, y_train, X_test, y_test):
	verbose, epochs, batch_size = 0, 15, 64
	n_timesteps = X_train.shape[1]
	n_features =  X_train.shape[2]
	n_outputs = y_train.shape[1]	
    # reshape data into time steps of sub-sequences
	n_steps, n_length = 4, 32
	X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))
	X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))
	# define model
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
	model.add(TimeDistributed(Dropout(0.5)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(100))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
	return accuracy, model


#-----------------load data---------------------
X_test_submission = np.load("X_test_kaggle.npy")
X_train = np.load("X_train_kaggle.npy")
y_train = np.loadtxt("y_train_final_kaggle.csv", dtype = np.str , delimiter = ',', usecols=(0,1), unpack=False)
y_train = y_train[:,1] 
print(X_train.shape)
X_train = np.swapaxes(X_train,1,2)
X_test_submission = np.swapaxes(X_test_submission,1,2)
print(X_train.shape)
#print(y_train.shape)
#y_train[:,0] = y_train[:,0].astype(np.int)


#--------------create an indexes from class names------
le = LabelEncoder()
le.fit(y_train)
classes = le.transform(y_train)
y_train = np.column_stack((y_train, classes)) # stack stings and their indexes


#--------------split the data for train ant tes--------------------------
#they write something about sklearn.model_selection.ShuffleSplit but I
#dont understand what they want, seems like they already shuffled the data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train[:,1], test_size=0.2)
print(X_train.shape)
print(y_train[:5])

y_train = to_categorical(y_train)
print(y_train[:10,:])
y_test = to_categorical(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


repeats = 20 # just repeating training hoping get better random result
best_model = Sequential()
model = Sequential()
best_score = 0
for r in range(repeats):
	score, model = evaluate_model(X_train, y_train, X_test, y_test)
	score = score * 100.0
	print('>#%d: %.3f' % (r+1, score))
	if score > best_score:
		best_model = model
		best_score = score

print(f'Best score: {best_score}')

#-------------create submission------------
# reshape data into time steps of sub-sequences
n_steps, n_length = 4, 32
n_features =  X_test_submission.shape[2]
X_test_submission = X_test_submission.reshape((X_test_submission.shape[0], n_steps, n_length, n_features))
y_pred = best_model.predict(X_test_submission)
y_pred = np.argmax(y_pred, axis=1) # convert probabilities to class index, which was one hote encoded before
print(y_pred[:10])
labels = list(le.inverse_transform(y_pred))


with open("submission.csv", "w") as fp:
    fp.write("# Id,Surface\n")
    for i, label in enumerate(labels):
        fp.write("%d,%s\n" % (i, label))










