import matplotlib.pyplot as plt
import tensorflow as tf
import random as rand
import numpy as np
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
    Takes an image and shapes it into a 1D vector
    :param images: array of images
    :return: list, flattened vector
    """

    num_images = images.shape[0]
    num_pixels = images.shape[1]**2
    return images.reshape(num_images, num_pixels)


def calculate_gaussian_vectors(x_train, y_train):
    """
    trains classifier model using naive Bayesian method
    :param x_train: flattened image vector
    :param y_train: label vector
    :return: mean and variance vectors
    """
    print("Calculating vectors...")

    # Add noise to data
    noise_x = np.random.normal(loc=0.0, scale=10, size=x_train.shape)
    x_train = x_train + noise_x

    num_classes = len(set(y_train))  # 10
    num_pixels = x_train.shape[1]  # 784

    # Initialize matrix of covariance and mean vectors
    mu = np.zeros((num_classes, num_pixels))
    sg = np.zeros((num_classes, num_pixels))

    for c in range(num_classes):
        # Search all samples belonging to class c
        class_samples = x_train[y_train == c]

        mu[c, :] = np.mean(class_samples, axis=0)
        sg[c, :] = np.var(class_samples, axis=0)

    return mu, sg


def compute_likelihood(x_test_flat, x_test, y_test, mu, sg):
    """
    shows accuracy and some images to see results of training
    :param x_test_flat: list, flattened image vector
    :param x_test: 28x28 image array
    :param y_test: vector of correct labels
    :param mu: mean vector
    :param sg: covariance vector
    :return: none
    """
    print("Computing likelihood...")

    predictions = []

    # Iterate over every class for every test sample
    for i in range(x_test_flat.shape[0]):
        likelihoods = []
        for c in range(len(set(y_test))):
            likelihood = np.sum(-0.5 * (np.log(2*np.pi)+np.log(sg[c]) +
                                        1 / sg[c] * (x_test_flat[i]-mu[c])**2))
            likelihoods.append(likelihood)

        # Append prediction with maximum likelihood estimation's corresponding
        # class
        predictions.append(np.argmax(likelihoods))

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

    # Calculate mean and covariance vectors
    mu, sg = calculate_gaussian_vectors(x_train, y_train)

    # Show some images and accuracy of model
    compute_likelihood(x_test_flat, x_test, y_test, mu, sg)


if __name__ == "__main__":
    main()
