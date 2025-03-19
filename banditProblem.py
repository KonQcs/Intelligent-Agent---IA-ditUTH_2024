import numpy as np
import matplotlib.pyplot as plt

class MultiArmedBandit:
    def __init__(self, m, n, std_dev):
        self.m = m  # number of machines
        self.n = n  # number of levers per machine
        self.std_dev = std_dev
        self.means = np.random.uniform(0, 10, size=(m, n))  # true mean rewards for each lever on each machine
        self.cov = std_dev ** 2 * np.eye(n)  # covariance matrix for each machine's levers

    def pull_lever(self, machine_idx, lever_idx):
        rewards = np.random.multivariate_normal(self.means[machine_idx], self.cov)
        return rewards[lever_idx]

class BanditSolver:
    def __init__(self, bandit, p):
        self.bandit = bandit
        self.p = p  # number of actions
        self.q_values = np.zeros((bandit.m, bandit.n))  # estimated rewards for each lever
        self.action_counts = np.zeros((bandit.m, bandit.n))  # counts of lever pulls
        self.total_reward = 0

    def epsilon_greedy(self, epsilon):
        rewards = []
        actions = []  # To track selected machine and lever
        for _ in range(self.p):
            if np.random.random() < epsilon:
                machine_idx = np.random.randint(0, self.bandit.m)
                lever_idx = np.random.randint(0, self.bandit.n)
            else:
                machine_idx, lever_idx = np.unravel_index(np.argmax(self.q_values), self.q_values.shape)

            reward = self.bandit.pull_lever(machine_idx, lever_idx)
            self.action_counts[machine_idx, lever_idx] += 1
            alpha = 1 / self.action_counts[machine_idx, lever_idx]
            self.q_values[machine_idx, lever_idx] += alpha * (reward - self.q_values[machine_idx, lever_idx])
            self.total_reward += reward
            rewards.append(self.total_reward)
            actions.append((machine_idx, lever_idx))
        return rewards, actions

    def softmax(self, tau):
        rewards = []
        actions = []  # To track selected machine and lever
        for _ in range(self.p):
            preferences = self.q_values / tau
            exp_preferences = np.exp(preferences - np.max(preferences))
            probabilities = exp_preferences / np.sum(exp_preferences)
            machine_idx, lever_idx = np.unravel_index(
                np.random.choice(np.arange(probabilities.size), p=probabilities.ravel()), self.q_values.shape
            )

            reward = self.bandit.pull_lever(machine_idx, lever_idx)
            self.action_counts[machine_idx, lever_idx] += 1
            alpha = 1 / self.action_counts[machine_idx, lever_idx]
            self.q_values[machine_idx, lever_idx] += alpha * (reward - self.q_values[machine_idx, lever_idx])
            self.total_reward += reward
            rewards.append(self.total_reward)
            actions.append((machine_idx, lever_idx))
        return rewards, actions

# Parameters
m = 5  # number of machines
n = 3  # number of levers per machine
p = 1000  # number of actions
std_dev = 1  # standard deviation for rewards

epsilon = 0.01  # exploration probability for epsilon-greedy
tau = 1  # temperature for softmax

# Initialize bandit and solver
bandit = MultiArmedBandit(m, n, std_dev)
solver = BanditSolver(bandit, p)

# Run epsilon-greedy and softmax methods
epsilon_greedy_rewards, epsilon_greedy_actions = solver.epsilon_greedy(epsilon)
print("Epsilon-Greedy Total Reward:", epsilon_greedy_rewards[-1])
print("Epsilon-Greedy Actions (machine, lever):", epsilon_greedy_actions[:p])

solver = BanditSolver(bandit, p)  # reset solver
softmax_rewards, softmax_actions = solver.softmax(tau)
print("Softmax Total Reward:", softmax_rewards[-1])
print("Softmax Actions (machine, lever):", softmax_actions[:p])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(epsilon_greedy_rewards, label="Epsilon-Greedy")
plt.plot(softmax_rewards, label="Softmax")
plt.xlabel("Actions")
plt.ylabel("Cumulative Reward")
plt.title("Comparison of Epsilon-Greedy and Softmax")
plt.legend()
plt.show()
