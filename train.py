import numpy as np
import random
from typing import Dict, List
import torch

from datetime import datetime

from collections import deque
from tqdm import tqdm

from catmouseenv import CatMouseEnv
from dqn import build_model, build_optimizer

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


def train_agent(config: Dict) -> List[float]:
    """
    Trains a DQN agent to play either the mouse or the cat (mouse should always go first?)
    """
    input_dim = config.get("input_dim")
    hidden_dims = config.get("hidden_dims")
    output_dim = config.get("output_dim")
    learning_rate = config.get("learning_rate", 1e-3)
    epochs = config.get("epochs", 1_000)
    batch_size = config.get("batch_size", 64)
    memory_size = config.get("memory_size", 2_000)
    gamma = config.get("gamma", 0.9)
    epsilon_start = config.get("epsilon_start", 0.3)
    epsilon_end = config.get("epsilon_end", 0.05)
    epsilon_decay = config.get("epsilon_decay", 0.995)
    sync_freq = config.get("sync_freq", 100)
    model_path = config.get("model_path", "model.pt")
    role = config.get("role", "mouse")
    mode = config.get("mode", "random")

    model, target_network = build_model(input_dim, hidden_dims, output_dim)
    optimizer = build_optimizer(model, learning_rate)
    loss_fn = torch.nn.MSELoss()

    replay_buffer = deque(maxlen=memory_size)
    losses = []
    j = 0  # Used to sync target network every sync_freq steps -> helps stabilize training

    for epoch in tqdm(range(epochs)):

        env = CatMouseEnv(mode=mode)
        state = torch.from_numpy(env.reset()).float().unsqueeze(0)
        done = False  # whether the current episode has terminated
        steps = 0

        while not done:

            j += 1
            steps += 1

            # Available actions depend on mouse's current neighbors
            neighbors = env.available_mouse_actions()
            action_space = {i: n for i, n in enumerate(neighbors)}

            # Action selection based on epsilon-greedy logic
            if np.random.random() < epsilon_start:
                # Choose by random
                action_idx = np.random.choice(list(action_space.keys()))
            else:
                qvals = model(state)
                action_idx = torch.argmax(qvals[:, : len(neighbors)]).item()

            action_node = action_space[action_idx]
            # Apply chosen action in the env and observe outcome
            next_state_np, reward, done = env.step(action_node)
            next_state = torch.from_numpy(next_state_np).float().unsqueeze(0)

            replay_buffer.append((state, action_idx, reward, next_state, done))
            state = next_state

            # Sample a random minibatch from the replay buffer and unpack the
            # components for training
            if len(replay_buffer) >= batch_size:

                mini_batch = random.sample(replay_buffer, batch_size)

                states, actions, rewards, next_states, dones = zip(*mini_batch)

                states = torch.cat(states)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float)
                next_states = torch.cat(next_states)
                dones = torch.tensor(dones, dtype=torch.float)

                # Extract predicted Q-values for taken actions
                qvals = model(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():

                    next_qvals = target_network(next_state).max(dim=1)[0]
                    targets = rewards + gamma * (1 - dones) * next_qvals

                loss = loss_fn(qvals, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if j % sync_freq == 0:
                    # Copy weights from model to target model (for stable Q-learning)
                    target_network.load_state_dict(model.state_dict())

                epsilon = max(epsilon_end, epsilon_start * epsilon_decay)
                if epoch % 500 == 0:
                    print(f"Epoch {epoch}, Role: {role}, Reward: {reward:.2f}, Loss: {loss.item()}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = model_path.replace(".pt", f"_{timestamp}.pt")

    torch.save(model.state_dict(), filename)
    print(f"Trained {role}, model saved to {filename}")
    return losses
