import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def manual_input(env):
    """
    lets user play the game with inputs 0-5
    :param env: game environment
    :return: None
    """

    while True:
        print(env.render())

        try:
            user_input = input("> ")

            if user_input == "e":
                break

            direction = int(user_input)

            if not 0 <= direction <= 5:
                print("Try again, 0-5")
                continue

        except ValueError:
            print("Try again, 0-5")
            continue

        except TypeError:
            print("Try again, 0-5")
            continue

        state, reward, done, truncated, info = env.step(direction)

        if done:
            return env


def eval_policy_better(env_, pi_, gamma_, t_max_, episodes_):
    v_pi_rep = np.empty(episodes_)  # N trials
    for e in range(episodes_):
        s_t = env_.reset()[0]
        v_pi = 0
        for t in range(t_max_):
            a_t = pi_[s_t]
            s_t, r_t, done, truncated, info = env_.step(a_t)
            v_pi += gamma_ ** t * r_t
            if done or truncated:
                break
        v_pi_rep[e] = v_pi
        env_.close()
    return np.mean(v_pi_rep), np.min(v_pi_rep), np.max(v_pi_rep), np.std(
        v_pi_rep)


def learn(env):
    aspace = env.action_space.n
    ospace = env.observation_space.n

    qtable = np.zeros((ospace, aspace))  # Taxi
    episodes = 15000  # More episodes for better learning
    interactions = 666  # max num of interactions per episode
    epsilon = 1.0  # Start with higher exploration
    decay_rate = 0.0001  # Epsilon decay per episode
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Reward decay rate
    debug = 1  # for non-slippery case to observe learning
    hist = []  # evaluation history

    # Main Q-learning loop
    for episode in range(episodes):
        state = env.reset()[0]
        step = 0
        done = False
        total_rewards = 0

        for interact in range(interactions):

            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, aspace)  # Explore
            else:
                action = np.argmax(qtable[state, :])  # Exploit

            # Observe
            new_state, reward, done, truncated, info = env.step(action)

            # Update Q-table
            qtable[state, action] = qtable[state, action] + alpha * (
                    reward + gamma * np.max(
                        qtable[new_state, :]) - qtable[state, action])

            # Our new state is state
            state = new_state

            # Check if terminated
            if done or truncated:
                break

        # Decay epsilon
        epsilon = max(0.0, epsilon - decay_rate)

        if epsilon == 0:
            alpha = 0.0001

        # Periodically evaluate and debug
        if episode % 500 == 0 or episode == 1:
            pi = np.argmax(qtable, axis=1)
            val_mean, val_min, val_max, val_std = eval_policy_better(env, pi,
                                                                     gamma,
                                                                     interactions,
                                                                     1000)
            hist.append([episode, val_mean, val_min, val_max, val_std])
            if debug:
                print(pi)
                print(f"{val_mean} // episode {episode}/{episodes}")

    pi_Q = np.argmax(qtable, axis=1)

    env.reset()

    return hist, pi_Q, gamma, env


def plot_performance(hist, env, pi_Q, gamma):
    hist = np.array(hist)
    print(hist.shape)

    plt.plot(hist[:, 0], hist[:, 1])
    plt.show()

    # Evaluate performance
    print(pi_Q)
    val_mean, val_min, val_max, val_std = eval_policy_better(env, pi_Q, gamma,
                                                             666, 1000)
    print(f'Value function mean {val_mean:.4f}, min {val_min:.4f} max '
          f'{val_max:.4f} and std {val_std:.4f}')


def main():
    env = gym.make("Taxi-v3", render_mode="ansi")
    env.reset()

    # For testing
    # env = manual_input(env)

    hist, pi_Q, gamma, env_trained = learn(env)

    plot_performance(hist, env_trained, pi_Q, gamma)


if __name__ == "__main__":
    main()
