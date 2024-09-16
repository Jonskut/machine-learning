import matplotlib.pyplot as plt
import tensorflow as tf
import random as rand
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


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
    print(f"accuracy {acc*100}%")


def main():
    # Original
    mnist = tf.keras.datasets.mnist
    # New
    # mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Print the size of training and test data
    print(f'x_train shape {x_train.shape}')
    print(f'y_train shape {y_train.shape}')
    print(f'x_test shape {x_test.shape}')
    print(f'y_test shape {y_test.shape}')

    x_train = flatten(x_train)
    x_test_flat = flatten(x_test)

    knn = train(x_train, y_train)
    show_accuracy(x_test_flat, x_test, y_test, knn)


if __name__ == "__main__":
    main()
