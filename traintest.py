import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Conv1D, GlobalMaxPooling1D, concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split

is_init = False
size = -1

label = []
dictionary = {}
c = 0

X = None  # Initialize X outside the loop
y_list = []  # Use a list to collect labels

for i in os.listdir():
    if i.split(".")[-1] == "npy" and not (i.split(".")[0] == "labels"):
        if not is_init:
            is_init = True
            X = np.load(i)
            size = X.shape[0]
            y_list.extend([i.split('.')[0]] * size)
        else:
            temp_X = np.load(i)
            size += temp_X.shape[0]
            X = np.concatenate((X, temp_X))
            y_list.extend([i.split('.')[0]] * temp_X.shape[0])

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1

if y_list:
    y = np.array(y_list).reshape(-1, 1)

    for i in range(y.shape[0]):
        y[i, 0] = dictionary[y[i, 0]]

    y = np.array(y, dtype="int32")
    y = to_categorical(y)

    X = X.reshape((X.shape[0], 1, X.shape[1]))

    X_new = X.copy()
    y_new = y.copy()

    counter = 0

    cnt = np.arange(X.shape[0])
    np.random.shuffle(cnt)

    for i in cnt[:min(X_new.shape[0], len(cnt))]:
        X_new[counter] = X[i]
        y_new[counter] = y[i]
        counter += 1
      
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

    ip = Input(shape=(X.shape[1], X.shape[2]))

    lstm_out = LSTM(64, activation='tanh')(ip)
    dense_out = Dense(128, activation='tanh')(lstm_out)
    dense_out = Dense(64, activation='tanh')(dense_out)
    cnn_out = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(ip)
    cnn_out = GlobalMaxPooling1D()(cnn_out)

    merged = concatenate([lstm_out, cnn_out])
    op = Dense(y.shape[1], activation='softmax')(merged)
    model = Model(inputs=ip, outputs=op)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(X_train, y_train, epochs=80)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f'Training Loss: {train_loss}, Training Accuracy: {train_accuracy}')

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Testing Loss: {test_loss}, Testing Accuracy: {test_accuracy}')

    model.save("model1.h5")
    np.save("labels1.npy", np.array(label))
else:
    print("No valid files found to process.")

