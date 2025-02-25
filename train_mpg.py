#!/usr/bin/env python3

import os
import numpy as np
from urllib.request import urlretrieve

from matplotlib import pyplot as plt

from mlp import (
    Layer, MultilayerPerceptron,
    Sigmoid, Tanh, Relu, Linear, Softplus, Mish,
    SquaredError, CrossEntropy
)


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
DATA_FILE = "auto-mpg.data"


def download_mpg_data():
    if not os.path.exists(DATA_FILE):
        print("Downloading Auto MPG dataset...")
        urlretrieve(DATA_URL, DATA_FILE)
        print("Download complete.")
    else:
        print("Auto MPG data file already exists. Skipping download.")

def load_and_preprocess_mpg():
    download_mpg_data()

    rows = []
    with open(DATA_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            mpg = parts[0]
            cylinders = parts[1]
            displacement = parts[2]
            horsepower = parts[3]
            weight = parts[4]
            acceleration = parts[5]
            model_year = parts[6]
            origin = parts[7]

            if horsepower == '?':
                horsepower = np.nan

            try:
                mpg = float(mpg)
                cylinders = float(cylinders)
                displacement = float(displacement)
                horsepower = float(horsepower) if horsepower != 'nan' else np.nan
                weight = float(weight)
                acceleration = float(acceleration)
                model_year = float(model_year)
                origin = float(origin)
            except ValueError:
                continue

            rows.append([mpg, cylinders, displacement, horsepower, weight,
                         acceleration, model_year, origin])

    data = np.array(rows, dtype=np.float32)
    horsepower_col = data[:, 3]
    mean_hp = np.nanmean(horsepower_col)
    inds_nan = np.isnan(horsepower_col)
    horsepower_col[inds_nan] = mean_hp
    data[:, 3] = horsepower_col

    np.random.shuffle(data)

    y = data[:, 0]
    X = data[:, 1:]

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std

    return X, y


def split_dataset(X, y):
    """
    Splits dataset into 70% train, 15% validation, 15% test.
    Returns (train_x, train_y, val_x, val_y, test_x, test_y).
    """
    N = X.shape[0]
    train_end = int(0.70 * N)
    val_end = int(0.85 * N)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_mlp(input_dim: int) -> MultilayerPerceptron:
    layer1 = Layer(input_dim, 64,Relu())
    layer2 = Layer(64,64, Relu())
    layer3 = Layer(64, 32, Relu(),0.1)
    # The final layer uses a Linear activation
    layer_out = Layer(32,     1,  Linear())

    # Build MLP
    mlp = MultilayerPerceptron(layers=(layer1, layer2, layer3, layer_out))
    return mlp


def main():
    X, y = load_and_preprocess_mpg()
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    model = build_mlp(input_dim=X_train.shape[1])
    loss_func = SquaredError()

    print("Starting training MLP on Auto MPG dataset...")
    train_losses, val_losses = model.train(
        train_x=X_train,
        train_y=y_train,
        val_x=X_val,
        val_y=y_val,
        loss_func=loss_func,
        learning_rate=1e-3,
        batch_size=32,
        epochs=100,
        rmsprop=False
    )

    y_pred_test = model.forward(X_test, training=False)
    test_loss = loss_func.loss(y_test, y_pred_test)
    print(np.round(y_pred_test[:10].reshape(1, 10), 1), np.round(y_test[:10].reshape(1, 10), 1))
    print(f"Final Test MSE (SquaredError) = {test_loss:.4f}")

    #plotting the training and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MPG MLP: Training & Validation Loss')
    plt.legend()
    plt.savefig('mpg_loss.png')

if __name__ == "__main__":
    main()