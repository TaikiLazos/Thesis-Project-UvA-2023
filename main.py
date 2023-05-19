import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from tqdm import tqdm

def simple_model(X_train, y_train):
    '''
    create single layer rnn model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    
    # create a model
    model = Sequential([
    tf.keras.layers.LSTM(16, input_shape=(n, X_train.shape[-1])),  # Define the RNN layer
    tf.keras.layers.Dense(n)  # Define the output layer
    ])

    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # fit the model
    history = model.fit(X_train, y_train, validation_split = 0.15, epochs=50, batch_size=64, verbose=0)

    return history, model

def plot_loss_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    TRAIN = False

    n = 5
    ver = "without"

    if TRAIN:
        # the data is normalized, and has everything except the latest 20% SHEL data
        x = np.load(f"Data/Training/n={n}/x_{ver}.npy")
        y = np.load(f"Data/Training/n={n}/y_{ver}.npy")

        if ver == 'with':
            x = np.array([i.T for i in x])
        else:
            # reshape
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        print("Training data shape")
        print(x.shape, y.shape)

        history, my_model = simple_model(x, y)
        my_model.save(f"Model/LSTM_model_n={n}_{ver}")
        plot_loss_curve(history)
    else:
        my_model = load_model(f'Model/LSTM_model_n={n}_{ver}')

    my_model.summary()

    ### Forecasting target company ###
    target = 'SHEL'
    
    # plot
    X_test = np.load(f"Data/Test/n={n}/x_{ver}_{target}.npy")
    Y_test = np.load(f"Data/Test/n={n}/y_{ver}_{target}.npy")

    if ver == 'with':
        X_test = np.array([i.T for i in X_test])

    dates = np.array(list(range(len(X_test))))

    X_test = np.reshape(X_test, (30, n, 1))
    y_predict = my_model.predict(X_test)
    forecast = np.array([])

    if ver == 'without':
        xi = np.reshape(X_test[0], (1, n))

    for _ in tqdm(range(len(X_test))):
        yi = np.mean(my_model.predict(np.reshape(xi, (1, n, 1))))
        forecast = np.append(forecast, yi)
        # pop the first index and append the yi to the last index
        xi =  np.delete(xi, [0], 1)
        yi = np.reshape(yi, (1, 1))
        xi = np.append(xi, yi, 1)

    plt.plot(dates,  Y_test[:,0], label = "Target")
    plt.plot(dates, np.mean(y_predict, axis = 1), label = "Prediction")
    plt.plot(dates, forecast, label = "Forecasting")

    plt.title(f"n={n}, {ver}")
    plt.legend()
    plt.show()