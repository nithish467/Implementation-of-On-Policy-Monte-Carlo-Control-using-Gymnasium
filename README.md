## NAME : NITHISHKUMAR S
## REG NO : 212223240109

# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

The experiment uses the Gymnasium FrozenLake-v1 environment, which is a 4×4 grid-world reinforcement learning problem.

Start State (S): Initial position of the agent.

Frozen Tiles (F): Safe tiles where the agent can move.

Holes (H): Unsafe tiles that terminate the episode.

Goal (G): Target state that provides a reward of 1.

The environment contains:

16 states (numbered 0–15).

4 actions in each state:

0: Left (L)

1: Down (D)

2: Right (R)

3: Up (U)

The environment is slippery (is_slippery=True), meaning the agent may not always move in the intended direction, making the learning process more challenging.



## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm

On-Policy Monte Carlo Control using Epsilon-Greedy Policy

1. Initialize the Q-table with zeros for all state-action pairs.
2. Set the learning parameters: learning rate (α), discount factor (γ), exploration rate (ϵ), and epsilon decay.
3. Generate a complete episode using the current epsilon-greedy policy.
4. Store each (state, action, reward) encountered during the episode.
5. Traverse the episode backward and compute the discounted return G.
6. Update the Q-value using the incremental Monte Carlo update rule:

Q(s,a)=Q(s,a)+α[G−Q(s,a)]

7. Reduce epsilon gradually to shift from exploration to exploitation.
8. Repeat the process for all training episodes.
9. Extract the learned policy using argmax(Q) and compute the state-value function using max(Q).
10. Display the Q-table, state-value function, learned policy, average reward, and learning curve.


## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Create Environment
# -------------------------------------------------

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Environment details
n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))
episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------


def epsilon_greedy_action(state, epsilon):
    """
    Selects an action using epsilon-greedy policy.
    """
    if np.random.random() < epsilon:
        return env.action_space.sample()      # Explore
    else:
        return np.argmax(Q[state])            # Exploit


# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(epsilon):
    """
    Generates one episode using the current epsilon-greedy policy.
    Returns a list of (state, action, reward).
    """

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):
        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode



# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for episode_num in range(num_episodes):

    # Generate one complete episode
    episode = generate_episode(epsilon)

    G = 0

    # Process episode backwards
    for state, action, reward in reversed(episode):

        G = gamma * G + reward

        # Incremental Monte Carlo update
        Q[state, action] = Q[state, action] + alpha * (G - Q[state, action])

    # Store episode reward
    episode_rewards.append(sum([step[2] for step in episode]))

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------

optimal_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)
    print("Name: NITHISHKUMAR S")
    print("Register Number: 212223240109")
    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(optimal_policy)

success_rate = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", success_rate)



# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500
moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Monte Carlo Control Learning Curve")
plt.grid(True)
plt.show()

env.close()




```

---

## Output



## For episode = 20000

### Final Q-table:

<img width="769" height="485" alt="image" src="https://github.com/user-attachments/assets/357e1f05-38c3-4f23-b739-5f3ddb3699fb" />


### Estimated State-Value Function and Learned Policy:


<img width="596" height="575" alt="image" src="https://github.com/user-attachments/assets/63ee4bcf-9408-4d3d-abde-ff5738098c3e" />





## For episode = 30000

### Final Q-table:


<img width="772" height="455" alt="image" src="https://github.com/user-attachments/assets/77d485b9-bd8f-4876-a34a-fcb7849df95b" />



### Estimated State-Value Function and Learned Policy:


<img width="509" height="591" alt="image" src="https://github.com/user-attachments/assets/48791bd0-34c4-4f0a-8074-358cece4fd2c" />




## Result

The On-Policy Monte Carlo Control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment.

When the number of training episodes was increased from 20,000 to 30,000, the agent showed improved learning performance:

The average reward over the last 1000 episodes increased from 0.039 to 0.071.

The learning curve exhibited higher and more frequent reward peaks.

The learned Q-values and state-value estimates improved, resulting in a better policy for navigating the FrozenLake environment despite the stochastic nature of the slippery surface.

## Inference

Increasing the training episodes from 20,000 to 30,000 improved the performance of the On-Policy Monte Carlo Control agent. The average reward over the last 1000 episodes increased from 0.039 to 0.071, and the learning curve showed higher and more frequent reward peaks. The Q-values and state-value estimates became stronger, indicating that additional training helped the agent learn a better policy for reaching the goal in the FrozenLake environment, although some fluctuations remained due to the slippery environment and epsilon-greedy exploration.




---

