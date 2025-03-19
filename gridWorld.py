import numpy as np
import matplotlib.pyplot as plt

def initialize_gridworld():
    """Αρχικοποιεί το GridWorld 4x4 και επιστρέφει τις απαραίτητες δομές."""
    grid_size = 4
    terminal_states = [(0, 0), (3, 3)]  # Οι τερματικές καταστάσεις
    rewards = -1  # Ανταμοιβή για όλες τις μεταβάσεις

    states = {(i, j): 0 for i in range(grid_size) for j in range(grid_size)}
    for t in terminal_states:
        states.pop(t)  # Αφαιρούμε τις τερματικές καταστάσεις

    actions = ['up', 'down', 'left', 'right']  # Πιθανές κινήσεις
    return states, actions, rewards, terminal_states

def is_valid_move(state, action, grid_size=4):
    """Ελέγχει αν μια κίνηση είναι έγκυρη."""
    i, j = state
    if action == 'up' and i > 0:
        return (i - 1, j)
    elif action == 'down' and i < grid_size - 1:
        return (i + 1, j)
    elif action == 'left' and j > 0:
        return (i, j - 1)
    elif action == 'right' and j < grid_size - 1:
        return (i, j + 1)
    return state  # Αν δεν είναι έγκυρη, επιστρέφει την ίδια κατάσταση

def policy_evaluation_two_tables(states, actions, rewards, terminal_states, gamma=1.0, theta=1e-4):
    """Επαναληπτική πολιτική αξιολόγηση με χρήση δύο πινάκων."""
    V = {s: 0 for s in states.keys()}  # Αρχικοποίηση συναρτήσεων τιμής
    while True:
        delta = 0
        new_V = V.copy()
        for state in states.keys():
            if state in terminal_states:
                continue
            v = 0
            for action in actions:
                next_state = is_valid_move(state, action)
                if next_state in terminal_states:
                    v += (1 / len(actions)) * rewards
                else:
                    v += (1 / len(actions)) * (rewards + gamma * V[next_state])
            new_V[state] = v
            delta = max(delta, abs(V[state] - v))
        V = new_V
        if delta < theta:
            break
    return V

def policy_evaluation_one_table(states, actions, rewards, terminal_states, gamma=1.0, theta=1e-4):
    """Επαναληπτική πολιτική αξιολόγηση με χρήση ενός πίνακα."""
    V = {s: 0 for s in states.keys()}  # Αρχικοποίηση συναρτήσεων τιμής
    while True:
        delta = 0
        for state in states.keys():
            if state in terminal_states:
                continue
            v = 0
            for action in actions:
                next_state = is_valid_move(state, action)
                if next_state in terminal_states:
                    v += (1 / len(actions)) * rewards
                else:
                    v += (1 / len(actions)) * (rewards + gamma * V[next_state])
            delta = max(delta, abs(V[state] - v))
            V[state] = v
        if delta < theta:
            break
    return V

def find_optimal_policy(states, actions, V, rewards, gamma=1.0):
    """Βρίσκει τη βέλτιστη πολιτική για κάθε κατάσταση."""
    policy = {}
    for state in states.keys():
        action_values = {}
        for action in actions:
            next_state = is_valid_move(state, action)
            if next_state in V:  # Έλεγχος αν υπάρχει στον πίνακα V
                action_values[action] = rewards + gamma * V[next_state]
            else:  # Αν είναι τερματική κατάσταση, λαμβάνει μόνο την ανταμοιβή
                action_values[action] = rewards
        best_action = max(action_values, key=action_values.get)
        policy[state] = best_action
    return policy

def visualize_grid(V, policy, grid_size=4):
    """Οπτικοποιεί τη συνάρτηση τιμής και την πολιτική."""
    grid = np.zeros((grid_size, grid_size))
    arrows = np.full((grid_size, grid_size), '', dtype=object)

    for (i, j), value in V.items():
        grid[i, j] = value
        if (i, j) in policy:
            if policy[(i, j)] == 'up':
                arrows[i, j] = '↑'
            elif policy[(i, j)] == 'down':
                arrows[i, j] = '↓'
            elif policy[(i, j)] == 'left':
                arrows[i, j] = '←'
            elif policy[(i, j)] == 'right':
                arrows[i, j] = '→'

    plt.figure(figsize=(8, 8))
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, grid_size - i - 1, f'{grid[i, j]:.1f}\n{arrows[i, j]}',
                     ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.grid(True)
    plt.title('Συνάρτηση Τιμής και Βέλτιστη Πολιτική')
    plt.show()

# Εκτέλεση της διαδικασίας
states, actions, rewards, terminal_states = initialize_gridworld()

print("Με χρήση δύο πινάκων:")
V_two_tables = policy_evaluation_two_tables(states, actions, rewards, terminal_states)
optimal_policy_two_tables = find_optimal_policy(states, actions, V_two_tables, rewards)
print("\nΣυνάρτηση Τιμής V(s):")
for state, value in V_two_tables.items():
    print(f"Κατάσταση {state}: {value:.2f}")
print("\nΒέλτιστη Πολιτική:")
for state, action in optimal_policy_two_tables.items():
    print(f"Κατάσταση {state}: {action}")
visualize_grid(V_two_tables, optimal_policy_two_tables)

print("\nΜε χρήση ενός πίνακα:")
V_one_table = policy_evaluation_one_table(states, actions, rewards, terminal_states)
optimal_policy_one_table = find_optimal_policy(states, actions, V_one_table, rewards)
print("\nΣυνάρτηση Τιμής V(s):")
for state, value in V_one_table.items():
    print(f"Κατάσταση {state}: {value:.2f}")
print("\nΒέλτιστη Πολιτική:")
for state, action in optimal_policy_one_table.items():
    print(f"Κατάσταση {state}: {action}")
visualize_grid(V_one_table, optimal_policy_one_table)
