import matplotlib.pyplot as plt
import tensorflow as tf
import random as rand
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import argparse


def class_acc(pred, gt):
    zeros_amount = 0
    # calculate difference of vectors
    difference = np.subtract(pred, gt)
    # amount of zeros is amount of correct guesses
    for num in difference:
        if num == 0:
            zeros_amount += 1

    # normalize
    accuracy = zeros_amount / len(gt)
    return accuracy


def flatten(images):
    return images.reshape(images.shape[0], 28 * 28)


def train(x_matrix, y_matrix):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_matrix, y_matrix)
    return knn


def show_accuracy(x_test_flat, x_test, y_test, knn):

    predictions = knn.predict(x_test_flat)

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


def load_dataset(dataset_name):
    if dataset_name == 'fashion':
        # Load Fashion MNIST dataset
        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.fashion_mnist.load_data()
        print("Using Fashion MNIST dataset")
    elif dataset_name == 'original':
        # Load original MNIST dataset
        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.mnist.load_data()
        print("Using Original MNIST dataset")
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
    knn = train(x_train, y_train)

    # Show some images and accuracy of model
    show_accuracy(x_test_flat, x_test, y_test, knn)


if __name__ == "__main__":
    main()
