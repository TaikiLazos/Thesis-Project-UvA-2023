import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def univariate_model(X_train, y_train):
    '''
    create single layer rnn model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    
    # create a model
    model = Sequential([
    tf.keras.layers.LSTM(16, input_shape=(n, X_train.shape[-1])),
    tf.keras.layers.Dense(n)  # Define the output layer
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # fit the model
    history = model.fit(X_train, y_train, validation_split = 0.15, epochs=50, batch_size=64, verbose=0)

    return history, model

def multivariate_model(X_train, y_train):
    '''
    create single layer rnn model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    
    # create a model
    model = Sequential([
    tf.keras.layers.LSTM(16, input_shape=(n, X_train.shape[-1]), return_sequences=True),
    tf.keras.layers.Dense(X_train.shape[-1])  # Define the output layer
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

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

def RMSE(y_true, y_hat):
    return mean_squared_error(y_true, y_hat, squared=False)

if __name__ == "__main__":
    # Set True if you want to train
    TRAIN = True
    
    # This sets the historical window size
    n = 5
    ver = "with" # with -> with commodity feature (multivariate model), without -> (univariate model)

    if TRAIN:
        # the data is normalized, and has everything except the latest 20% SHEL data
        x = np.load(f"Data/Training/n={n}/x_{ver}.npy")
        y = np.load(f"Data/Training/n={n}/y_{ver}.npy")

        if ver == 'with':
            x = np.array([i.T for i in x])
            y = np.array([i.T for i in y])
            print("Training data shape")
            print(x.shape, y.shape)

            history, my_model = multivariate_model(x, y)
        else:
            # reshape
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))
            print("Training data shape")
            print(x.shape, y.shape)

            history, my_model = univariate_model(x, y)

        my_model.save(f"Model/LSTM_model_n={n}_{ver}")
        plot_loss_curve(history)
    else:
        my_model = load_model(f'Model/LSTM_model_n={n}_{ver}')

    my_model.summary()

    ### Forecasting target company ###
    target = 'SHEL'
    
    # plot
    X_test = np.load(f"Data/Test/n={n}/x_{ver}_{target}.npy")
    Y_test = np.load(f"Data/Test/n=5/y_without_{target}.npy")

    forecast = np.array([])
    
    if ver == 'with':
        X_test = np.array([i.T for i in X_test])
        Y_test = np.array([i.T for i in Y_test])
        xi = np.reshape(X_test[0], (1, n, 2))

        for _ in tqdm(range(len(X_test))):
            yi = np.mean(my_model.predict(xi), axis=1)
            forecast = np.append(forecast, yi[0][0])
            # pop the first index and append the yi to the last index
            xi =  np.delete(xi, [0], 1)
            xi = np.append(xi[0], yi, 0)
            xi = np.reshape(xi, (1, n, 2))
    else:
        X_test = np.reshape(X_test, (30, n, 1))
        xi = np.reshape(X_test[0], (1, n))

        for _ in tqdm(range(len(X_test))):
            yi = np.mean(my_model.predict(np.reshape(xi, (1,n,1))))
            forecast = np.append(forecast, yi)
            # pop the first index and append the yi to the last index
            xi =  np.delete(xi, [0], 1)
            yi = np.reshape(yi, (1, 1))
            xi = np.append(xi, yi, 1)

    dates = np.array(list(range(len(X_test))))
    # y_predict = my_model.predict(X_test)
 
    # print('prediction:', RMSE(y_predict[:,  0], Y_test[:,  0]))
    print('forecasting:',RMSE(Y_test[:,  0], forecast))

    plt.plot(dates, Y_test[:, 0], label = "Target")
    # plt.plot(dates, y_predict[:,  0], label = "Prediction")
    plt.plot(dates, forecast, label = "Forecasting")

    plt.title(f"n={n}, {ver}")
    plt.legend()
    plt.show()