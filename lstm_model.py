import tensorflow as tf
from keras.models import Sequential
import matplotlib.pyplot as plt

def lstm(x, y, M):
    """
    A LSTM model.
    It has 6 Keras layers, 
    LSTM(32) -> Dropout -> LSTM(18) -> Dropout -> Dense -> Reformat
    """
    # parameter setting
    epochs = 50
    batch_size = 32
    
    # building a model
    # my_model = Sequential(
    #     [
    #         tf.keras.layers.LSTM(
    #             32,
    #             input_shape=x.shape[-2:], # (N, # of features)
    #             return_sequences=False,
    #             name='LSTM32'
    #         ),
    #         tf.keras.layers.Dropout(0.1),
    #         tf.keras.layers.RepeatVector(M),
    #         tf.keras.layers.LSTM(18, return_sequences=True, name='LSTM18'),
    #         tf.keras.layers.Dropout(0.1),
    #         tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(x.shape[-1]), name='Dense'),
    #     ]
    # )
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
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(M * x.shape[-1]),
            tf.keras.layers.Reshape((M, x.shape[-1])),
            tf.keras.layers.Dense(x.shape[-1])
        ]
    )


    # my_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    my_model.compile(optimizer="RMSprop", loss="mean_squared_error")

    my_model.summary()

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


# my_model = Sequential(
#         [
#             tf.keras.layers.LSTM(
#                 32,
#                 input_shape=x.shape[-2:], # (N, # of features)
#                 return_sequences=True,
#             ),
#             tf.keras.layers.Dropout(0.1),
#             tf.keras.layers.LSTM(18, return_sequences=True),
#             tf.keras.layers.Dropout(0.1),
#             tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(x.shape[-1])),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(M * x.shape[-1]),
#             tf.keras.layers.Reshape((M, x.shape[-1])),
#             tf.keras.layers.Dense(8)
#         ]
#     )

# my_model.summary()