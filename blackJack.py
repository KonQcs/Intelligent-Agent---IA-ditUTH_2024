import numpy as np
import random

# Define constants for the game
HIT = 0
STICK = 1

# Initialize parameters
states = []  # List of states: (player_sum, dealer_card, usable_ace)
for player_sum in range(4, 22):  # Extend range to include all possible sums
    for dealer_card in range(1, 11):
        for usable_ace in [True, False]:
            states.append((player_sum, dealer_card, usable_ace))

# Initialize Q-value function and returns dictionary
Q = {state: {HIT: 0, STICK: 0} for state in states}
returns = {state: {HIT: [], STICK: []} for state in states}

# Initialize policy: stick only if player_sum >= 20, otherwise hit
policy = {state: STICK if state[0] >= 20 else HIT for state in states}

def draw_card():
    """Draw a card from the deck."""
    card = random.randint(1, 13)
    return min(card, 10)  # Face cards count as 10

def initialize_game():
    """Initialize a new game."""
    player_hand = [draw_card(), draw_card()]
    dealer_hand = [draw_card(), draw_card()]
    return player_hand, dealer_hand

def hand_value(hand):
    """Calculate the value of a hand."""
    value = sum(hand)
    usable_ace = 1 in hand and value + 10 <= 21
    if usable_ace:
        value += 10
    return value, usable_ace

def is_bust(value):
    """Check if a hand is bust."""
    return value > 21

def play_dealer(dealer_hand):
    """Play the dealer's turn based on the fixed strategy."""
    while True:
        value, _ = hand_value(dealer_hand)
        if value >= 17:
            break
        dealer_hand.append(draw_card())
    return hand_value(dealer_hand)[0]

def episode():
    """Simulate one episode of Blackjack."""
    player_hand, dealer_hand = initialize_game()
    player_value, usable_ace = hand_value(player_hand)
    dealer_card = dealer_hand[0]

    # Initial state
    state = (player_value, dealer_card, usable_ace)
    trajectory = []

    # Player's turn
    while True:
        if state[0] >= 20:  # Force stick if player_sum >= 20
            action = STICK
        else:
            action = policy[state]

        trajectory.append((state, action))
        if action == STICK:
            break
        player_hand.append(draw_card())
        player_value, usable_ace = hand_value(player_hand)
        if is_bust(player_value):
            return trajectory, -1  # Player loses
        state = (player_value, dealer_card, usable_ace)

    # Dealer's turn
    dealer_value = play_dealer(dealer_hand)
    if is_bust(dealer_value):
        return trajectory, 1  # Dealer busts, player wins

    # Determine winner
    if player_value > dealer_value:
        return trajectory, 1
    elif player_value < dealer_value:
        return trajectory, -1
    else:
        return trajectory, 0

# Monte Carlo Exploring Starts
num_episodes = 10000  # Reduced for faster execution
for episode_idx in range(1, num_episodes + 1):
    trajectory, reward = episode()

    # Update Q-values
    for state, action in trajectory:
        returns[state][action].append(reward)
        Q[state][action] = np.mean(returns[state][action])

    # Update policy
    for state in states:
        if state[0] >= 20:  # Force stick if player_sum >= 20
            policy[state] = STICK
        else:
            policy[state] = max(Q[state], key=Q[state].get)

    # Print progress every 1000 episodes
    if episode_idx % 1000 == 0:
        print(f"Completed {episode_idx}/{num_episodes} episodes.")

# Display final policy and Q-values
print("Final Policy:")
for state, action in policy.items():
    print(f"State {state}: {'HIT' if action == HIT else 'STICK'}")

print("\nQ-Values:")
for state in states:
    print(f"State {state}: HIT={Q[state][HIT]:.2f}, STICK={Q[state][STICK]:.2f}")
