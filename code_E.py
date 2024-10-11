import numpy as np

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

# Print results
print(f"Out of {men} male passengers {men_alive} survived and {men-men_alive} died")
print(f"    (surveillance rate {men_rate:.2f}%)")
print(f"Out of {women} female passengers {women_alive} survived and {women-women_alive} died")
print(f"    (surveillance rate {women_rate:.2f}%)")