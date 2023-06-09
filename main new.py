# 1. Preprocessing -> preprocessing.py (Adds features and reformats in desirable numpy array)
# 2. Differential Privacy -> LS-LSTM shows improvement so why not -> main.py
# 3. Algorithm X -> I need to do something here like TDM & TDC equivalent. -> main.py
# 4. LSTM -> LSTM(32) -> dropout -> LSTM(16) -> dropout -> reformat to (M, # features) -> main.py
# 5. Prediction & Forecasting

import numpy as np
# import tensorflow_privacy
import tensorflow as tf
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import pandas as pd
import data_process

def lstm(x, y):
    """
    A LSTM model.
    It has 6 Keras layers, 
    LSTM(32) -> Dropout -> LSTM(18) -> Dropout -> Dense -> Reformat
    """
    # parameter setting
    epochs = 50
    batch_size = 250

    # # For differntial privacy
    # l2_norm_clip = 1.5
    # noise_multiplier = 1.3
    # num_microbatches = 250
    # learning_rate = 0.25

    # if batch_size % num_microbatches != 0:
    #     raise ValueError('Batch size should be an integer multiple of the number of microbatches')

    # optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    #     l2_norm_clip=l2_norm_clip,
    #     noise_multiplier=noise_multiplier,
    #     num_microbatches=num_microbatches,
    #     learning_rate=learning_rate)

    # loss = tf.keras.losses.CategoricalCrossentropy(
    #     from_logits=True, reduction=tf.losses.Reduction.NONE)
    
    # building a model
    my_model = Sequential(
        [
            tf.keras.layers.LSTM(
                32,
                input_shape=x.shape[-2:], # (N, # of features)
                return_sequences=True,
            ),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.LSTM(18, return_sequences=True),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(x.shape[-1])),
            tf.keras.layers.Lambda(lambda x: x[:, -M:, :])  # Adjust output shape
        ]
    )


    # my_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    my_model.compile(optimizer="RMSprop", loss="mean_squared_error", metrics=['accuracy'])

    my_model.summary()

    # Shuffle the data?

    # fit the model
    hist = my_model.fit(
        x, y, validation_split=0.15, epochs=epochs, batch_size=batch_size, verbose=0
    )

    return hist, my_model

def plot_loss_curve(h):
    """
    Plotting loss curve of a model.
    """
    plt.plot(h.history["loss"])
    plt.plot(h.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    # Global settings
    TRAIN = True # Train model or not?
    N = 5 # historical input window size N = 5, 7, 10, 14
    M = 3 # historical output window size, only 3 is available right now
    VER = "with" # with -> with commodity feature, without -> without commodity feature

    if TRAIN:
        # Load the training data
        x_train = np.load(f"Data/Training/n={N}/x_{VER}.npy", allow_pickle=True)
        y_train = np.load(f"Data/Training/n={N}/y_{VER}.npy", allow_pickle=True)

        print("Training Data Loading...")
        print(f"x = {x_train.shape}, y = {y_train.shape}")
        
        # Train the model
        history, model = lstm(x_train, y_train)

        # if you want you can plot the loss curve
        # plot_loss_curve(history)

        # Save the model
        model.save("Model/Test_model")
    else:
        model = load_model("Model/Test_model")
        model.summary()

    # Prediction
    # Load test data
    commodity = 'CL=F'
    company = 'XOM'
    x_train = np.load(f"Data/Training/n={N}/x_{VER}.npy", allow_pickle=True)
    y_train = np.load(f"Data/Training/n={N}/y_{VER}.npy", allow_pickle=True)

    test_df = pd.read_csv(f"Data/Test/{commodity}/n={N}/x_{VER}_{company}.csv")
    tdf = test_df.iloc[:, 2:]
    x_test, y_test = data_process.create_x_y(tdf, N, M)

    result = model.predict(x_test)

    # We only care about the adjusted close price
    y_hat = result[::2, 0, 0]
    y_true = y_test[::2, 0, 0]

    days = np.array(np.arange(len(y_true)))
    plt.plot(days,y_true)
    plt.plot(days, y_hat)
    plt.savefig('result.png')
