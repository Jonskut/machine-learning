import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = np.loadtxt('train.dat', delimiter=' ', encoding='utf8')

# Total number of passengers
total_passengers = data.shape[0]

# Number of survivors and deaths
men = np.sum(data[:, 2] == 0)
women = np.sum(data[:, 2] == 1)

# AI USED FOR CODE; PARSING AND READING FILE

# Survivors by gender [AI used for help (ChatGPT)]
men_alive = np.sum((data[:, 1] == 1) & (data[:, 2] == 0))
women_alive = np.sum((data[:, 1] == 1) & (data[:, 2] == 1))

men_rate = (men_alive/men) * 100
women_rate = (women_alive/women) * 100

# AI USED FOR CODE; PARSING AND READING FILE

# Fares by dead and alive men and women AI used (ChatGPT)
fares_dead_men = data[(data[:, 1] == 0) & (data[:, 2] == 0), 4]
fares_alive_men = data[(data[:, 1] == 1) & (data[:, 2] == 0), 4]

fares_dead_women = data[(data[:, 1] == 0) & (data[:, 2] == 1), 4]
fares_alive_women = data[(data[:, 1] == 1) & (data[:, 2] == 1), 4]

# Create a figure with two subplots ChatGPT AI used to help with plotting
# two figures at the same time
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the histograms for Men
axs[0].hist(fares_alive_men, alpha=0.6, color='blue', edgecolor='black', label='Survived')
axs[0].hist(fares_dead_men, alpha=0.6, color='red', edgecolor='black', label='Died')
axs[0].set_xlabel('Fare Amount')
axs[0].set_ylabel('Number of Male Passengers')
axs[0].set_title('Comparison of Fares Between Survived and Dead Men')
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Plotting the histograms for Women
axs[1].hist(fares_alive_women, alpha=0.6, color='blue', edgecolor='black', label='Survived')
axs[1].hist(fares_dead_women, alpha=0.6, color='red', edgecolor='black', label='Died')
axs[1].set_xlabel('Fare Amount')
axs[1].set_ylabel('Number of Female Passengers')
axs[1].set_title('Comparison of Fares Between Survived and Dead Women')
axs[1].legend(loc='upper right')
axs[1].grid(True)

# Display the plot
plt.show()
