import matplotlib.pyplot as plt
import tensorflow as tf
import random as rand
import numpy as np


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

ground_truth = []
prediction = []
iterations = 0

print(range(x_test.shape[0]))

for i in range(x_test.shape[0]):
    # Show some images randomly
    if rand.random() > 0.995:
        iterations += 1
        ran = rand.randint(0, 9)
        plt.figure(1)
        plt.clf()
        plt.imshow(x_test[i], cmap='gray_r')
        plt.title(f"Image {i} label num {y_test[i]} predicted {ran}")
        ground_truth.append(y_test[i])
        prediction.append(ran)
        plt.pause(0.1)

acc = class_acc(prediction, ground_truth)
print(f"accuracy {acc*100}%, {iterations} iterations")