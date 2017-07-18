import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

from scipy.linalg import toeplitz

s = np.sin(2 * np.pi * 0.125 * np.arange(20))
print s
S = np.fliplr(toeplitz(np.r_[s[-1], np.zeros(s.shape[0] - 2)], s[::-1]))
print S[:5, :3]

X_train = S[:-1, :3][:, :, np.newaxis]
Y_train = S[:-1, 3]
print X_train.shape, Y_train.shape
print X_train[:2]

np.random.seed(0)
model = Sequential()
model.add(SimpleRNN(10, input_dim=1, input_length=3))
model.add(Dense(1))
model.compile(loss='mse', optimizer='sgd')

history = model.fit(X_train, Y_train, nb_epoch=100, verbose=0)

plt.plot(Y_train, 'ro-', label="target")
plt.plot(model.predict(X_train[:,:,:]), 'bs-', label="output")
plt.xlim(-0.5, 20.5)
plt.ylim(-1.1, 1.1)
plt.legend()
plt.title("After training")
plt.show()