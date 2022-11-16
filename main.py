import os
import datetime
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers


def load_data(file_path):
    with open('./data/train.npy', 'rb') as f:
        input = np.load(f)
        target = np.load(f)
    return input, target

def normalize_data_per_row(data):
    fdata = np.zeros(data.shape)

    # TODO. Complete.
    for i in range(data.shape[0]):
        dmin = data[i].min()
        dmax = data[i].max()
        fdata[i] = (data[i] - dmin) / float(dmax)
    return fdata

def splite_data(input, target, percent):
    assert input.shape[0] == target.shape[0], \
        "Number of inputs and targets do not match ({} vs {})".format(input.shape[0], target.shape[0])
    indices = range(input.shape[0])
    np.random.shuffle(list(indices))
    num_train = int(input.shape[0] * percent)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    return input[train_indices, :], target[train_indices, :], input[test_indices, :], target[test_indices, :]

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape = input_shape))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def train_model(model, train_input, train_target, val_input, val_target,
                epochs, learning_rate, batch_size, logs_dir):
    # compile the model: define optimizer, loss, and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                 loss='binary_crossentropy',
                 metrics=['binary_accuracy'])

    norm_train_input = normalize_data_per_row(train_input)
    norm_val_input = normalize_data_per_row(val_input)

    # TODO - Create callbacks for saving checkpoints and visualizing loss on TensorBoard
    # tensorboard callback
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)

    # save checkpoint callback
    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir,'best_weights.h5'),
                                                            monitor='binary_accuracy',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            save_weights_only=False,
                                                            mode='auto',
                                                            save_freq=1)
    # do training for the specified number of epochs and with the given batch size
    # TODO - Add callbacks to fit funciton
    model.fit(norm_train_input, train_target, epochs=epochs, batch_size=batch_size,
             validation_data=(norm_val_input, val_target),
             callbacks=[tbCallBack, checkpointCallBack])


def main(input_file, batch_size, epochs, lr, percent, logs_dir):
    print("Importing images...")
    input, target = load_data(input_file)
    print(input.shape)
    print(target.shape)

    assert(input.shape[0] == target.shape[0], "shape not match")
    train_input, train_target, val_input, val_target = splite_data(input, target, percent)

    model = build_model(train_input.shape[1:])
    # train the model
    print("\n\nTRAINING...")
    train_model(model, train_input, train_target, val_input, val_target,
                epochs=epochs, learning_rate=lr, batch_size=batch_size, logs_dir=logs_dir)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="number of epochs for training",
                        type=int, default=20)
    parser.add_argument("--batch_size", help="batch size used for training",
                        type=int, default=128)
    parser.add_argument("--lr", help="learning rate for training",
                        type=float, default=1e-2)
    parser.add_argument("--percent", help="percent of training data to use for validation",
                        type=float, default=0.8)
    parser.add_argument("--input", help="input file",
                        type=str, required=True)
    parser.add_argument("--logs_dir", help="logs directory",
                        type=str, default="")
    args = parser.parse_args()

    if len(args.logs_dir) == 0: # parameter was not specified
        args.logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))

    if not os.path.isdir(args.logs_dir):
        os.makedirs(args.logs_dir)

    # run the main function
    main(args.input, args.batch_size, args.epochs, args.lr, args.percent, args.logs_dir)