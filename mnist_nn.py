import matplotlib.pyplot as plt
import tensorflow as tf
import random as rand
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import argparse


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


def train_knn(x_train, y_train):
    """
    trains 1nn-model with scikit learn
    :param x_train: flattened image vector
    :param y_train: training labels
    :return: trained 1nn-model parameters
    """
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)
    return knn


def show_accuracy(x_test_flat, x_test, y_test, knn):
    """
    shows accuracy and some images to see results of training
    :param x_test_flat: list, flattened image vector
    :param x_test: 28x28 image array
    :param y_test: vector of correct labels
    :param knn: trained nearest-neighbor model
    :return: none
    """
    predictions = knn.predict(x_test_flat)

    for i in range(x_test_flat.shape[0]):
        # Show some images randomly
        if rand.random() > 0.9985:

            plt.figure(1)
            plt.clf()
            plt.imshow(x_test[i], cmap='gray_r')
            plt.title(f"Image {i} label num "
                      f"{y_test[i]} predicted {predictions[i]}")

            plt.pause(1)

    acc = class_acc(predictions, y_test)
    print(f"Classification accuracy is {acc:.2f}")


def load_dataset(dataset_name):
    """
    Loads the dataset from parsed CLI arguments
    :param dataset_name: str, origin or fashion
    :return: training and test dataset
    """
    if dataset_name == 'fashion':
        # Load Fashion MNIST dataset
        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.fashion_mnist.load_data()
        print("Using fashion MNIST dataset")
    elif dataset_name == 'original':
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

    # Train model
    knn = train_knn(x_train, y_train)

    # Show some images and accuracy of model
    show_accuracy(x_test_flat, x_test, y_test, knn)


if __name__ == "__main__":
    main()
