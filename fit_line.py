import matplotlib.pyplot as plt
import numpy as np


def my_linfit(x_points, y_points):
    """
    Linear solver
    :param x_points: list, of x coordinates
    :param y_points: list, of y coordinates
    :return: float, a and b of linear equation
    """
    n = len(x_points)
    a_num = sum(x_points * y_points) - (sum(x_points) * sum(y_points)) / n
    a_den = sum(x_points ** 2) - ((sum(x_points)) ** 2) / n

    a = a_num / a_den

    b = (sum(y_points) - (a * sum(x_points))) / n

    return a, b


def on_click(event):
    """
    Mouse event handler
    :param event: button click on mouse (left or right click)
    :return: none
    """
    global x, y
    global is_line

    # Add point
    if event.button == 1 and not is_line:
        x.append(event.xdata)
        y.append(event.ydata)
        plt.plot(event.xdata, event.ydata, 'ko')
        plt.draw()

    # Draw line
    elif event.button == 3 and not is_line and x and y:
        x_numpy = np.array(x)
        y_numpy = np.array(y)

        # Calculate fit
        a, b = my_linfit(x_numpy, y_numpy)

        # Plot the fitted line
        xp = np.linspace(min(x), max(x), 100)
        plt.plot(xp, a * xp + b, 'r-')

        plt.draw()
        is_line = True

        ax.set_title("Right click to reset view")

    # Reset view and points
    elif event.button == 3:
        plt.cla()
        ax.set_title("Click to add points, right-click to show fit line")

        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        is_line = False
        x = []
        y = []

        plt.draw()


def main():
    """
    Program for linear regression
    :return: none
    """


# List of points
x = []
y = []

# Flag for resetting view
is_line = False

fig, ax = plt.subplots()
ax.set_title("Click to add points, right-click to show fit line")

ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])

cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

if __name__ == "main":
    main()
