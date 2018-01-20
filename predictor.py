from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Activation


def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    print(x_train.shape)
    print(y_train.shape)
    model.add(LSTM(3, input_shape=(263, 1)))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)