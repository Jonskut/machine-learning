import matplotlib.pyplot as plt
import tensorflow as tf
import random as rand
import numpy as np
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,\
    Flatten, Dense, Dropout
import keras
from keras.utils import to_categorical
from keras import Input


def class_acc(pred, gt):
    """
    calculates classifier accuracy
    :param pred: list, predicted class
    :param gt: list, actual class
    :return: float, accuracy from 0-1
    """
    zeros_amount = 0
    # Calculate difference of vectors
    difference = np.subtract(pred, gt)
    # Amount of zeros is amount of correct guesses
    for num in difference:
        if num == 0:
            zeros_amount += 1

    # Normalize
    accuracy = zeros_amount / len(gt)
    return accuracy


def flatten(images):
    """
    Takes a 28x28 image and shapes it into a 1D vector
    :param images: array of images
    :return: list, flattened vector
    """
    return images.reshape(images.shape[0], 28 * 28)


def hotify(y_train):
    """
    turns training data to one-hot encoding
    :param y_train: training data
    :return: one-hot encoded data
    """
    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=10)

    return y_train


def create_model(x_train, y_train):
    """
    Creates NN model, trains it and shows loss progression
    :param x_train: MNIST training data
    :param y_train: MNIST training labels
    :return: NN model
    """
    # Initialize model and input shape
    model = Sequential()
    model.add(Input(shape=(784,)))

    # Optimizer
    sgd = keras.optimizers.SGD(learning_rate=0.1)

    # Fully connected layers
    model.add(Dense(128, activation='sigmoid'))

    # Output layer
    model.add(Dense(10, activation='sigmoid'))

    model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

    # Train the model
    tr_hist = model.fit(x_train, y_train, epochs=30, verbose=1)

    # Plot training loss over epochs
    plt.plot(tr_hist.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    return model


def show_accuracy(x_test_flat, x_test, y_test, model):
    """
    shows accuracy and some images to see results of training
    :param x_test_flat: list, flattened image vector
    :param x_test: 28x28 image array
    :param y_test: vector of correct labels
    :param model: trained neural network model
    :return: none
    """
    predict = model.predict(x_test_flat)
    predictions = np.argmax(predict, axis=1)

    for i in range(x_test_flat.shape[0]):
        # Show some images randomly
        if rand.random() > 0.999:

            plt.figure(1)
            plt.clf()
            plt.imshow(x_test[i], cmap='gray_r')
            plt.title(f"Image {i} label num "
                      f"{y_test[i]} predicted {predictions[i]}")

            plt.pause(1)

    acc = class_acc(predictions, y_test)
    print(f"Classification accuracy is {acc:.2f}")


def load_dataset(dataset_name="original"):
    """
    Loads the dataset from parsed CLI arguments
    :param dataset_name: str, origin or fashion
    :return: training and test dataset
    """
    if dataset_name == "fashion":
        # Load Fashion MNIST dataset
        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.fashion_mnist.load_data()
        print("Using fashion MNIST dataset")
    elif dataset_name == "original":
        # Load original MNIST dataset
        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.mnist.load_data()
        print("Using original MNIST dataset")
    else:
        raise ValueError(
            "Invalid dataset name. Choose 'fashion' or 'original'.")

    return (x_train, y_train), (x_test, y_test)


def main():
    parser = argparse.ArgumentParser(
        prog="mnist_nn.py",
        description="Image classifier using MNIST dataset"
    )

    parser.add_argument(
        "dataset",
        type=str,
        choices=["original", "fashion"]
    )

    args = parser.parse_args()

    # Load the chosen dataset
    (x_train, y_train), (x_test, y_test) = load_dataset(args.dataset)

    # Flatten image matrices
    x_train = flatten(x_train)
    x_test_flat = flatten(x_test)

    # "Hotify" y-vectors
    y_train = hotify(y_train)

    # Train model
    model = create_model(x_train, y_train)

    # Show some images and accuracy of model
    show_accuracy(x_test_flat, x_test, y_test, model)


if __name__ == "__main__":
    main()
