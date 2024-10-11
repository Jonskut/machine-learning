import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = np.loadtxt('train.dat', delimiter=' ', encoding='utf8')

# Total number of passengers
total_passengers = data.shape[0]

# Number of survivors and deaths
survived = np.sum(data[:, 1] == 1)
died = np.sum(data[:, 1] == 0)

# AI USED FOR CODE; PARSING AND READING FILE

died_percentage = (died / total_passengers) * 100
survived_percentage = (survived / total_passengers) * 100

# Output the results
print(f"Out of {total_passengers} passengers {survived}"
      f" ({survived_percentage:.2f}%) survived and {died} ({died_percentage:.2f}%) died")

# Fares by dead and alive, AI used
fares_dead = data[data[:, 1] == 0, 4]
fares_alive = data[data[:, 1] == 1, 4]

# Plotting the histograms for both groups
plt.figure(figsize=(10, 6))

# Histogram for survived passengers
plt.hist(fares_alive, alpha=0.6, color='blue', edgecolor='black', label='Survived')

# Histogram for dead passengers
plt.hist(fares_dead, alpha=0.6, color='red', edgecolor='black', label='Died')

# Add labels and title
plt.xlabel('Fare Amount')
plt.ylabel('Number of Passengers')
plt.title('Comparison of Fares Between Survived and Dead Passengers')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()