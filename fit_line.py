import matplotlib.pyplot as plt
import numpy as np


# Linear Solver
def my_linfit(x, y):
    n = len(x)
    a_num = sum(x * y) - (sum(x) * sum(y)) / n
    a_den = sum(x ** 2) - ((sum(x))**2) / n

    a = a_num / a_den

    b = (sum(y) - (a * sum(x))) / n

    return a, b


# Event handler for mouse clicks
def on_click(event):
    global x, y
    global is_line

    # Add point
    if event.button == 1 and not is_line:
        x.append(event.xdata)
        y.append(event.ydata)
        plt.plot(event.xdata, event.ydata, 'ko')  # Plot the point
        plt.draw()  # Update the plot

    # Draw line
    elif event.button == 3 and not is_line and x and y:
        x_numpy = np.array(x)
        y_numpy = np.array(y)

        # Calculate the fit
        a, b = my_linfit(x_numpy, y_numpy)

        # Plot the fitted line
        xp = np.linspace(min(x), max(x), 100)
        plt.plot(xp, a * xp + b, 'r-')

        plt.draw()  # Update the plot
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


# Main
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
